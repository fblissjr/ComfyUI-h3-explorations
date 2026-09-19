# The sister checkouts: what each one is good for

last updated: 2026-09-19 (section "What moved by 2026-09-19" added); 2026-09-15 (section "The streaming references: TaoMate" added; "What moved by 2026-09-11" added and the two comfy-kitchen rows corrected 2026-09-11; "What moved by 2026-09-10" added 2026-09-10; the tables are otherwise the 2026-08-28 read)

`coderef/` holds the reference implementations. `ls -l coderef/` is the list of
what is currently on disk — some symlinks, some real clones — and this page is
what each one is *for*: what it implements, what has actually been compared
against it, and what it is not evidence of.

**Written by a person.**

**Revisions are an observation point, not a pin.** Every one below was read on
the date in the header. A sister checkout moves under you; re-read before
quoting. Two of the H3 engines moved during the 2026-08-28 comparison pass
itself.

**Do not import Python from `coderef/`.** CLAUDE.md's rule, with the escaped
instance that earns it: requiring the clone and prepending it to `sys.path` is
how a bench script made itself unrunnable on a box that had the wheel and no
checkout. Use the clone for sources you cannot import; import the rest.

**Searching it.** More than half of `coderef/` is symlinks, so use `find -L`
and `grep -r --dereference-recursive`, or a search answers about a minority of
it. Two references live in this repo instead: `bench/_sol_attn_reference.py`
is the Sol-Attn reference for what can be imported, and
`vendor/sol_attn_minimax.py` is a read-only reference node that is not loaded.

---

## The four H3 implementations worth comparing against

These are the ones that implement MiniMax H3 end to end. All four were compared
against our node chain on 2026-08-28; the findings live in
[`../custom_node_gaps.md`](../custom_node_gaps.md), which this page routes to
rather than restating.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `sglang` | `803b4fb31c` | **the vendor's own serving path.** The closest thing to ground truth for what MiniMax intended | you need to know what the release actually does at a stage |
| `LightX2V` | `5169278f` | inference engine; **origin of the SLA work and the Turbo LoRAs we load** | anything about SLA, DMD step distillation, offload, or what a LoRA was distilled under |
| `DiffSynth-Studio` | `102fe99` | model library with a native H3 pipeline, its own converters and a LoRA path | you need a second opinion on a state-dict namespace or a converter |
| `diffusers` | `9f7aee482` | model library with a native H3 pipeline, a named H3 scheduler, and a conversion script | you need the canonical tensor namespace, or a clean statement of the sampler |

Two owner documents already exist for the first of these and are the authority
over anything here: [`../research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md)
for what sglang does stage by stage, and
[`../research/sglang_comparison.md`](../research/sglang_comparison.md) for what
its serving path does that we do not.

### What the 2026-08-28 pass established about them

Recorded here because it is a property of the *references*, not of our code:

- **Three-against-one is a real signal and it fired twice.** sglang, DiffSynth
  and diffusers agree on feeding one prepared reference tensor to both towers,
  and on running the video VAE more precisely than we do. Both are open.
- **Agreement is broad and worth banking.** The reference label rules, the
  encoder layer, the absence of a chat template, and the VAE normalisation
  statistics all agree across implementations. When four implementations agree,
  a fifth reading is not the cheapest next step.
- **Neither model library is independent evidence about the seven markers.**
  Both inherit the release tokenizer without touching the ids in code. Two more
  implementations is not two more votes.
- **No engine implements PDD.** diffusers, LightX2V, DiffSynth and sglang were
  each searched. See [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md).

---

## The ComfyUI-side references

| checkout | revision read | what it is |
|---|---|---|
| `comfy-kitchen` | `7490d87` | **the build source since 2026-09-08**: the owner's fork, based on an upstream tag and carrying only the `blk_cnt` commits (`vendor/rebuild_kernel.sh` defaults `SRC` to it). Since 2026-09-11 it is one checkout on `h3-build`, the one branch we build (ComfyUI's pinned tag plus `blk_cnt`); the fork holds only `main`, `h3-build`, PR 168's `sol-blk-cnt-pr`, and `archive/*` tags for retired builds that records cite. The script's "The build branch" header owns that layout, and `vendor/rebuild_kernel.sh --check` prints the base tag, our commits and the submodule pins. Import the installed Python rather than requiring the clone. *Corrected 2026-09-11: this row said "the upstream of the above", and later that day that the build branch was `sol-blk-cnt-<pin>`.* |
| `comfy-kitchen-kijai` | `bd3fc78` | kijai's fork, **read-only**, renamed from `comfy-kitchen-sol` on 2026-09-08. Its `.cu` files ship in no wheel, so `morton.md` quotes it by path under the old name. *Corrected 2026-09-11: this row named the clone `comfy-kitchen-sol` and said its built branch was installed; neither has been true since 2026-09-08.* |
| `ComfyUI-UtilsCollection` | `5bac35b` | a third-party pack with its own PDD path. Two of our guards were **adopted from it** |
| `Minimax-H3-Turbo` | `02e26d5` | the vendor README that publishes the distilled sigma grid `bench/check_distill_grid.py` grades against — a grid from the vendor, not one we computed |
| `sage-fork` | `56a5be4` | our SageAttention fork |
| `SLA` | `7db4039` | the sparse top-k attention reference |
| `TurboDiffusion` | `e3d6136` | step-distillation reference |

---

## The upstream and infrastructure clones

Not H3 implementations. Listed so nobody mistakes one for a comparison target.

| checkout | what it is | what it is not |
|---|---|---|
| `MiniMax-H3` | the release repository | not a runnable pipeline for our purposes |
| `MiniMax-Music3` | a different model | not H3 |
| `transformers`, `vllm`, `llm-compressor` | encoder-side and quantisation infrastructure | say nothing about the DiT |
| `Model-Optimizer` | NVIDIA's PTQ/QAT toolkit and its recipe catalogue (`modelopt_recipes/ptq.md`), read 2026-09-19 at `b311c054d`. It ships a `model_type/qwen3_vl/ptq` pair that puts FP8 on the Qwen3-VL **vision** branch's Linear layers, including the deepstack mergers, and deliberately leaves the patch embedding and the vision attention BMM operands high precision. Two things it is useful for: a scope precedent for the vision tower our encoder keeps entirely in BF16 (the header of `h3_config.ENCODER_INT8`), and its named calibration levers (MSE weight search, local Hessian, NVFP4 activation headroom, SmoothQuant alpha — its Gemma override picks the same 0.5 our `channel_balance.py` defaults to) | not something to run: it exports TensorRT-LLM checkpoints, which ComfyUI does not load, its NVFP4 schemes need Blackwell, and calibrating our own encoder is a closed lane (`docs/roadmap.md`, "Closed lanes", 2026-08-27) |
| `triton`, `flashinfer`, `nanobind` | kernel infrastructure | |
| `Sana`, `h3-turbo-eval` | adjacent research | |

---

## What moved by 2026-09-10

Read on 2026-09-10 by fetch, at the revisions named here; the tables above keep
their 2026-08-28 revisions. Sol-side movement (Sana's Sol-H3, sglang's SubBlock
work) lives in [`../sol_upstream.md`](../sol_upstream.md).

- **`vllm-omni`** (`ffcaaa943`; an H3 serving engine not in the tables above).
  `af74a5a15` (#7062) derives a LightX2V Turbo file's sampler contract from its
  filename alone -- 768p files at video shift 6, 544p at 12, audio 3 -- and
  scales by the file's declared alpha over rank, falling back to alpha 8 only
  for a file that declares none (`vllm_omni/diffusion/models/minimax_h3/lora.py`).
  That matches ComfyUI's alpha/rank scaling, confirmed on the v1.1 and v1.2 768p
  files, whose `.alpha` tensors carry the declared value. By its rule the two
  8-step v1.0 768p files on disk (fl2v and ref2v) would run at 6/3;
  `bench/check_distill_settings.py` deliberately classifies neither, and no
  shipped graph loads them. `30d6a0b4e` (#7191) pins cuDNN flags around the
  keyframe encode: [`../open_experiments.md`](../open_experiments.md) #30.
  `715b8b874` (#6720) validates the text-conditioning handoff and changes no
  numerics.
- **`LightX2V`** (`fabad304`). `95e9b86b` (#1503) adds a persistent AdaLN cache,
  an offline-built table keyed on the exact float32 bits of each timestep and
  enabled across its H3 DMD configs. `35d4aaa8` (#1464) wraps kitchen's
  `int8_linear` in a `torch.library.custom_op` so torch.compile can trace INT8
  ConvRot; core calls it unwrapped (`comfy/ops.py`), and this repo never
  compiles. `e1088278` (#1506) adds optional FP8 Conv3D modes to the video VAE
  encoder, off by default. `fabad304` (#1511) redefined H3 `infer_steps` from
  sigma grid points to evaluations, with the same behaviour;
  `bench/check_distill_settings.py` reads those configs.
- **`flashinfer`**: `4fa42525` adds a MiniMax-H3 MXFP8 pre-attention kernel for
  SM100a/SM103a and `01587699` documents H3 run parameters. Neither runs on this
  card.
- **`DiffSynth-Studio`**: `ce9f454` (#1678) adds an optional H3 training
  adapter. **`diffusers`**: `d30c748f5` touches H3 LoRA tests only.
- **`Minimax-H3-Turbo`**: the clone is still at `02e26d5`. Its README has rows
  for the v1.0 768p 4-step and 8-step files only, and neither it nor the HF
  model README carries a card for v1.1 or v1.2. The HF repo added a v1.1 768p
  fp8 file on 2026-09-10.
- **`ComfyUI-UtilsCollection`** (`d6a9600`). H3 reference nodes landed
  2026-09-05 to 09-09: save, load and apply VAE-encoded reference latents; a
  reference-video component that resamples to H3's frame rate and the audio
  VAE's sample rate and pads its audio's end to the audio VAE's hop; a media config whose "even keyframes" mode
  places reference-video chunks as keyframes; a "VLM guide" that splices a
  separately encoded Qwen forward in front of the prompt text; and an encode
  cache keyed on the encoder object, its patches and the token bytes, which
  refuses a mismatched file and lives in ComfyUI's temp directory, so it does
  not outlast a restart. Its `d1921ae` reverted per-section encoding after
  reported generation distortion and caches the joint encode only.
- **Hugging Face, not cloned**: community FastH3 conversions (NVFP4 rotated,
  GGUF, a dense-datafree ComfyUI file); `junchaoh-cs/SolarWM-H3-33B` is gated
  and was not read.

---

## What moved by 2026-09-11

Read on 2026-09-11 at the revisions named here. comfy-kitchen, core's Sol
node and the ComfyUI Sol packs live in [`../sol_upstream.md`](../sol_upstream.md)
(section "Comfy-Org/comfy-kitchen, as of 2026-09-11"); sglang lives in
[`../research/sglang_comparison.md`](../research/sglang_comparison.md)
(section "Fifth read").

- **`DiffSynth-Studio`** (`32ef37e`). `013296e` and `84f93fc` predate the
  2026-09-10 section and were missed by it; the rest landed since.
  - `013296e` (#1659) loads alibaba-pai's Fun ControlNet Union: a second
    stack of DiT blocks whose outputs are added to the main hidden stream
    after a fixed set of main blocks (`diffsynth/models/minimax_h3_controlnet.py`;
    the hook is `control_hints` in `diffsynth/models/minimax_h3_dit.py`, the
    block set is in `diffsynth/configs/model_configs.py`). Core has its own
    loader (`comfy/ldm/minimax/controlnet.py`, reached from
    `comfy_extras/nodes_model_patch.py`). Nothing in this repo wires a
    ControlNet, so this is a second implementation to read if one is ever
    wired, not a gap.
  - `84f93fc` (#1655) adds `--audio_loss_weight` to its H3 training script.
    Training only.
  - `ce9f454` (#1678), with the README entry that came with it: the
    "Training Adapter" is a DeCFG LoRA in FL2VA and Ref2VA versions, trained
    on a self-generated dataset, for fine-tuning on top of the CFG-distilled
    base, plus two toy LoRAs trained through it. Training only, and it agrees
    with [`../prompting.md`](../prompting.md) that guidance is CFG-distilled.
  - `f7b7db9` (#1684) loads the third-party Singularity ref2va v1.3 INT8
    files, full and pruned, as ComfyUI-format checkpoints quantized for
    comfy-kitchen's INT8 W8A8. Its converter only strips the
    `model.diffusion_model.` prefix
    (`diffsynth/utils/state_dict_converters/minimax_h3_dit.py`), and the two
    entries leave different modules unquantized (`minimax_h3_series` in
    `diffsynth/configs/model_configs.py`). That corroborates the fourth
    sglang read: such a file loads in ComfyUI as it is. No graph here loads
    it.
  - `50e5efb` (#1681) announces
    [DiffSynth-ComfyUI](https://github.com/modelscope/DiffSynth-ComfyUI), a
    ComfyUI pack that runs DiffSynth's pipelines, created 2026-09-04. Its
    `example_workflows/` has H3 fl2va, ref2va and pruned-NF4 graphs. Not read
    past its file list.
  - Everything else is other models: LTX-2.5 (`a98c6d4`), YuE2 (`32ef37e`),
    DiffSynth-Music (`a8fc4a6`).
- **`sglang`** (`593c7a900d`): nothing reaches H3; see the fifth read.
- **`comfy-kitchen-kijai`** (fetched 2026-09-11). `minimax_vae` has not moved
  since `a63ca28` (2026-09-09); what its PR means for the DiT is in
  [`../open_experiments.md`](../open_experiments.md) #28. `w4a8_gemv` holds
  one commit of its own, `1caa605` (2026-08-24: a single-row W4A8 GEMV and an
  unrelated `gated_delta` op), and only merges from main since. A W4A8 H3
  file sits in this install's model directory, and neither
  `workflows/h3_config.py`, the generator nor any shipped graph names it.
- **Upstream PRs, not cloned**: Comfy-Org/ComfyUI #16239 and the kitchen and
  ComfyUI-pack PRs around it are read in `sol_upstream.md`. *2026-09-19:
  16239 and kitchen 171 were closed unmerged on 2026-09-16; see that page's
  section "comfy-kitchen and core, 2026-09-19".*

---

## What moved by 2026-09-19

Read on 2026-09-19 by fetch, from each clone's upstream branch, against the
revision its last section recorded. The list of commits for any clone is
`git log <recorded>..origin/main` inside it (Sana: `origin/sol-engine`).
comfy-kitchen, kijai's fork and core's PRs live in
[`../sol_upstream.md`](../sol_upstream.md), section "comfy-kitchen and core,
2026-09-19"; sglang's sixth read lives in
[`../research/sglang_comparison.md`](../research/sglang_comparison.md).

**Nothing below changes what runs on this card, and nothing triggers the
adopt-upstream rule.** Only sglang, core's node and comfy-kitchen can trigger
it, and none of the three moved a default this repo differs on.

- **`vllm-omni`** (`ffcaaa943` to `fa506e0fe`), the most H3 activity of any
  clone.
  - **Its ComfyUI "workflows" do not run H3 in ComfyUI.**
    `coderef/vllm-omni/apps/ComfyUI-vLLM-Omni` is a ComfyUI pack whose nodes
    send a request to a vllm-omni server
    (`coderef/vllm-omni/apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md` says
    ComfyUI loads no H3 weights). Its example graphs (t2v, ref2va, upscale,
    FastH3) are therefore templates for that server's sampler, which is
    deterministic Euler on a shifted linear schedule with no CFG
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/time_request.py`).
    They differ from our shipped graphs on base step count, sampler, frame
    count, the canvas used with a reference video, and the 768p Turbo LoRA's
    version, step count and strength; the templates are in
    `coderef/vllm-omni/apps/ComfyUI-vLLM-Omni/example_workflows/` and ours are
    `workflows/h3_config.py`. vllm-omni is not one of the rule's upstreams, so
    any of these is an ordinary judgement call. They agree with ours on
    shift, fps, the base canvas and no CFG.
  - `cb439f3e8` (#7610) loads `FastVideo/FastVideo-FastH3-8-Step-V2`, a full
    distilled DiT in Diffusers layout, on its own sigma ladder and shift
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py`).
    ComfyUI cannot load that layout as shipped. If one is ever converted,
    `bench/check_distill_settings.py` would classify the graph as base (it
    keys on LoRA filenames) and demand the base shift, which V2 does not use.
  - `507cb1d83` (#7535) moves its H3 VSA into the model. Same cube tiling and
    gated coarse branch as `vsa_attention.py`; it keeps a fixed count of
    video tiles where ours keeps a fraction (`h3_config.VSA_KEEP_PERCENT`).
  - `5d3e6dc1b` (#7693) rounds the DiT's AdaLN RMSNorm output to bf16 before
    scale and shift, on SM90 only
    (`coderef/vllm-omni/vllm_omni/diffusion/layers/indexed_modulation.py::_use_hopper_bf16_affine_semantics`).
    Core already applies scale and shift in the input dtype
    (`comfy/ldm/minimax/model.py`, `_mod_scale_shift`).
  - `3d952d133` (#7281) gives a reference video's soundtrack and a
    standalone reference audio separate duration budgets. Core has no
    reference-audio budget to conflate. `43b8de9b0` (#7167) fuses q/k
    RMSNorm and RoPE in the **video VAE**, not the DiT, on SM90 and newer.
    The rest is multi-GPU serving, NPU docs and a Music 3 node.
- **`LightX2V`** (`fabad304` to `52161985`). No H3 default changed on NVIDIA.
  `3acec9e8`'s fused DiT QKV, norm and RoPE path is enabled only in Intel XPU
  configs; `6214d38a` loads the Qwen3-VL encoder's vision tower at init
  (memory and load time, not numerics); `8335bb48` lets the fl2av variant
  take ref2av requests; the rest is XPU VAE attention, a multi-GPU VAE tiling
  fix, an FP8 VAE decoder converter fix and shared host offload.
- **`Sana`** (`sol-engine`, `757d902` to `ca26dbd`). `ca26dbd` (#507) adds
  HyperFlow, a third-party 8-step adapter for H3 that conditions each step on
  the interval it covers, so it swaps the DiT's time embedder for a two-time
  one (`coderef/Sana/models/minimax_h3/HyperFlow/hyperflow_h3/embedder.py`).
  No ComfyUI H3 model has that embedder, so it is not a LoRA-load away; its
  grid is not one `bench/check_distill_grid.py` grades; its runtime requires
  eight GPUs. `bb60499` fuses unmerged LoRA branches into consumer kernels,
  multi-GPU.
- **`flashinfer`** (`01587699` to `dc04f50c`). H3 BF16 pre-attention for
  SM100a and SM103a (`561f5af7`, `604da4ff`) and an SM120 Sage block-sparse
  kernel (`6a84331e`). Nothing runs on SM89.
- **`ComfyUI-UtilsCollection`** (`d6a9600` to `1d5b202`), mostly clip
  continuation.
  - **How it continues a clip.** It saves the previous clip's last decoded
    frames and audio
    (`coderef/ComfyUI-UtilsCollection/helpers/model_helpers.py::save_minimax_h3_clip_continuation_media`).
    It re-encodes them through its own encoder, which writes core's
    `minimax_keyframes` directly: the first and last tail frames become
    single-frame DiT keyframes, and the tail audio becomes guide-audio rows,
    not a reference. On the encoder side, the first tail frame goes in as a
    Picture and the rest as a Video block
    (`coderef/ComfyUI-UtilsCollection/helpers/encoder_helpers.py::execute_advanced_minimax_h3_image_to_video`).
    After rendering, an accumulator finds the re-rendered overlap by frame
    and audio similarity and cuts at the best seam.
  - **A lead, not a recommendation.** This is the decode-and-re-encode arm
    that [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 5 lists
    and never ran. Ours carries the sampled latent tail instead
    (`audio_freeze.py::MiniMaxH3FreezeAudioWindow`). Theirs also carries
    generated audio forward, which ours has no path for. No closed lane
    covers continuation.
  - **Two departures from the release's keyframe rules:** a keyframe inside
    the clip, and a keyframe that carries audio.
  - **Encoder pixel bounds.** `e25109c` patches the Qwen3-VL encoder's
    still-image pixel bounds to the release's
    (`coderef/ComfyUI-UtilsCollection/helpers/minimax_h3_preprocessing_helpers.py`).
    That is a runtime workaround for
    [`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) gaps 3 and 4,
    for stills only.
- **`TaoMate-H3`** (`ccc1a70` to `b933d8e`): README only. It now names the
  released adapter T2AV and lists FL2AV and Ref2AV as coming. Adapter, grid
  and audio regime are unchanged.
- **`DiffSynth-Studio`** (`32ef37e` to `c458cb4`): an audio loader fix for
  another model and a README. **`diffusers`** (`d30c748f5` to `a3e0b8ec2`):
  no H3 path touched.
- **Unmoved:** `Minimax-H3-Turbo` (`02e26d5`), `MiniMax-H3` (`d21241f`),
  `TurboDiffusion` (`e3d6136`), `TaoMate-LTX` (`136d890`).
- **Three clones on disk that the tables above do not list:**
  `comfyui_dagthomas` (a third-party H3 prompt-writer pack with chain
  renderers), `h3-the-transformation-engine` (the sister prompt compiler) and
  `ComfyUI-H3-Quant` (the owner's pack published from `standalone/h3_quant`).
  None is an H3 implementation to compare against.
- **Not read:** the infrastructure clones (`transformers`, `vllm`,
  `llm-compressor`, `triton`) past a commit-message search for H3 and
  Qwen3-VL since 2026-09-10, which found test fixes and a CPU and
  pipeline-parallel fix to vllm's Qwen3-VL; and the bodies of the multi-GPU
  and XPU commits named above.

---

## The streaming references: TaoMate

Read 2026-09-15, at the revisions named here.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `TaoMate-H3` | `ccc1a70` | TaoLiveAIGC's streaming runtime for H3. A 3-step LoRA on the FL2VA partition, run over each 5-second request in causal chunks: the chunk's video attends to the prompt, to a clean K/V cache and to itself, and a sigma-zero forward after each chunk commits its K/V. The cache keeps the first chunk's video as a sink and the two most recent chunks (`coderef/TaoMate-H3/src/taomate_h3/streaming/cache.py::CleanAVKVCache.retain_sink_and_recent_commits`). It accepts only 4 or 8 GPUs under TP2 with Ulysses and requires FlashAttention-3 (`coderef/TaoMate-H3/src/taomate_h3/config.py::DirectRunConfig`, `coderef/TaoMate-H3/src/taomate_h3/streaming/attention_hook.py`), so it does not run on this box. The adapter converts to a ComfyUI LoRA by rename: `bench/convert_taomate_lora.py`, record `bench/results/2026-09-15_taomate_lora_conversion.json` | you need the adapter's distilled sigma grid (`coderef/TaoMate-H3/src/taomate_h3/model/pipeline.py`, `DISTILLED_STATE_INDICES` at its two shifts), how a KV-cached causal H3 is wired, or an H3 team's own audio-freeze regime |
| `TaoMate-LTX` | `136d890` | the same group's LTX 2.3 system and the code for their paper (arXiv 2607.24359): learned persistent memory, reference-aware FiLM, a pyramid K/V retention policy, stage-parallel inference | the paper's mechanisms. Not evidence about H3 |

What the H3 checkout is not evidence of:

- **The paper's memory.** The H3 adapter holds LoRA factors on the existing
  attention and MLP linears and nothing else (the converter's
  `adapter_inventory` refuses any other tensor). The H3 runtime's only
  appearance mechanism is an untrained per-channel renormalisation of each
  chunk to the first chunk's statistics
  (`coderef/TaoMate-H3/src/taomate_h3/streaming/runtime.py::H3StreamingRuntime._renorm_clean_video_rows`).
- **Streamed or distilled audio.** `coderef/TaoMate-H3/src/taomate_h3/teacher.py` runs the base
  model with no adapter over each request, and the streaming loop overwrites
  the adapter's audio with those states after every step, then asserts the
  published audio equals the base result
  (`coderef/TaoMate-H3/src/taomate_h3/streaming/runtime.py::_base_audio_teacher_step_callback`
  and the guard after the phase loop). The README's speed table excludes that
  pass by its own note. What the adapter does to audio in a ComfyUI graph is
  outside anything its authors run. This is the audio-freeze regime:
  [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 5.
- **Anything about a single-card graph.** The README's timings are its own
  base runtime on its own node.

The community ComfyUI copies: kijai's `minimax_h3_taomate_3step_lora_avg_rank_19_bf16`
is a per-module truncated SVD of this adapter, in the same qkv and SwiGLU
layout. How much of each delta it keeps is in the record's `comparison`,
measured against the full-rank conversion. How the two differ in use, and the
plan for running the adapter in a graph, is
[`../h3_taomate.md`](../h3_taomate.md). The distilled grid is copied into
`workflows/h3_config.py`'s TaoMate block with this revision as its pointer, so
nothing that converts or builds needs this checkout.

"ComfyUI does not support KV chunking", as reported on 2026-09-15, is right
in substance: core's KV-cached causal sampler
(`comfy/k_diffusion/sampling.py::sample_ar_video`) requires a diffusion model
exposing `init_kv_caches`, which only Causal-Wan does, and core's H3 model
runs one unmasked attention over the whole packed sequence
([`../research/technique_transfer.md`](../research/technique_transfer.md),
fact 2). The "attention chunking" nodes in third-party H3 packs slice queries
to cut peak VRAM and carry no cache between chunks.

---

## The trap this page exists to prevent

**Two models live in this repo, and the words for their parts do not
disambiguate the stage.** "Attention" and "capture" each name something at the
DiT *and* at the Qwen3-VL encoder. A fact about one is not a fact about the
other, and three instances of carrying a DiT-side fact to an encoder-side
conclusion happened in a single day. CLAUDE.md holds the full rule; the tell is
always a type or a module prefix, never the vocabulary of the claim.

This applies with extra force to the sister checkouts, because a clone gives you
a confident, well-written source for the wrong stage.
