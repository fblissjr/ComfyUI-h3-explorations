# The sister checkouts: what each one is good for

last updated: 2026-10-10 (a row for the SAM-family checkouts and `sapiens2`, pointing at `meta_perception_models.md`); 2026-10-09 (section "What moved by 2026-10-09" added, with the masking and video-to-video cross-check; rows for `MaskVidExperiments` and `ComfyUI-NKD-Basic-Tools`); 2026-10-05 (section "What moved by 2026-10-05" added, with a dated note on the 2026-10-02 UtilsCollection bullet it supersedes; the `pdmd` row and the contract example, on PDMD's retirement); 2026-10-02 (section "What moved by 2026-10-02" added; a ComfyUI-H3-AudioRefine row; dated notes on the LightX2V row, the 2026-09-25 INT8 VAE sentence, and a correction to the 2026-09-25 AdaLN rounding claim); 2026-10-01 (the PDMD trainer: a row in the ComfyUI-side table, and its inference script added to what counts as a contract); 2026-09-27 (the TaoMate section trimmed to the checkouts after the lane was removed); 2026-09-26 (section "A distill's reference is its trainer's contract" added; the FastH3 V2 note under 2026-09-19 extended); 2026-09-25 (section "What moved by 2026-09-25" added; the PDD line and the vllm-omni #7693 bullet corrected in place); 2026-09-19 (section "What moved by 2026-09-19" added); 2026-09-15 (section "The streaming references: TaoMate" added; "What moved by 2026-09-11" added and the two comfy-kitchen rows corrected 2026-09-11; "What moved by 2026-09-10" added 2026-09-10; the tables are otherwise the 2026-08-28 read)

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

## A distill's reference is its trainer's contract

The owner's rule, 2026-09-26. A distilled checkpoint or LoRA is run the way
its trainer says to run it: the sampling contract the trainer ships.

**What counts as the contract:**
- a settings file in the release (FastVideo's `fastvideo_inference.json`);
- a release's merge script and schedule (FlashGen's `merge_lora_ckpt.py` and
  `base_schedule`);
- a serving engine's validator that pins those values
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py`
  checks FastH3 V2 against its file);
- a trainer's own inference script and the scheduler it pins. The one
  instance here was PDMD's `worker/run_a10.py`, retired with its lane on
  2026-10-05.

**The contract outranks a downstream template**, ComfyUI's included. A
template is a claim to test against the contract. When they differ, the
contract is the default and the template is an arm. CLAUDE.md's adopt-upstream
rule covers defaults that sglang and ComfyUI agree on. This rule covers a
distill's sampling, where the trainer is the one source with the answer.

**The instance that earned it:** FastH3 V2 was first built to ComfyUI's
template. That template departs from FastVideo's contract on the sampler, the
VSA kept fraction and a dense warm-up. The contract arms rendered a different
and more complete take (`../../bench/results/2026-09-26_fasth3_contract_s1.md`).
FlashGen was already on its contract's schedule
(`../research/2026-09-26_flashgen.md`).

---

## The four H3 implementations worth comparing against

These are the ones that implement MiniMax H3 end to end. All four were compared
against our node chain on 2026-08-28; the findings live in
[`../custom_node_gaps.md`](../custom_node_gaps.md), which this page routes to
rather than restating.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `sglang` | `803b4fb31c` | **the vendor's own serving path.** The closest thing to ground truth for what MiniMax intended | you need to know what the release actually does at a stage |
| `LightX2V` | `5169278f` | inference engine; **origin of the SLA work and the Turbo LoRAs we load** *(2026-10-02: both lanes are closed, turbo on 2026-09-26 and SLA on 2026-09-27, `docs/roadmap.md` "Closed lanes"; no shipped graph loads a Turbo LoRA and `bench/check_distill_settings.py` fails one that does)* | anything about SLA, DMD step distillation, offload, or what a LoRA was distilled under |
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
- **No engine implemented PDD as of 2026-08-28.** diffusers, LightX2V, DiffSynth and sglang were
  each searched. See [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md).
  *Corrected 2026-09-25: sglang has implemented it since `973fb44471`
  (#40568, 2026-09-23). This line used to say "No engine implements PDD".
  [`../research/sglang_comparison.md`](../research/sglang_comparison.md),
  "Seventh read", has the comparison.*

---

## The ComfyUI-side references

| checkout | revision read | what it is |
|---|---|---|
| `comfy-kitchen` | `7490d87` | **the build source since 2026-09-08**: the owner's fork (`vendor/rebuild_kernel.sh` finds its build branch by name). Since 2026-09-27 we build and install `h3-frontier`, upstream main plus our Sol commits, checked out in the clone itself (the workspace checkout this symlink points to), so what this checkout shows is what is installed. `h3-build`, the retired tag-based line that every record before 2026-09-27 cites, stays as a branch. The script's "The build branch" header owns that layout, and `vendor/rebuild_kernel.sh --check` prints the upstream base, how far upstream main has moved past it, our commits and the submodule pins. *Corrected 2026-09-27: this row said the build was ComfyUI's pinned tag plus our commits, on `h3-build`.* Import the installed Python rather than requiring the clone. *Corrected 2026-09-11: this row said "the upstream of the above", and later that day that the build branch was `sol-blk-cnt-<pin>`.* |
| `comfy-kitchen-kijai` | `bd3fc78` | kijai's fork, **read-only**, renamed from `comfy-kitchen-sol` on 2026-09-08. Its `.cu` files ship in no wheel, so `morton.md` quotes it by path under the old name. *Corrected 2026-09-11: this row named the clone `comfy-kitchen-sol` and said its built branch was installed; neither has been true since 2026-09-08.* |
| `ComfyUI-UtilsCollection` | `5bac35b` | a third-party pack with its own PDD path. Two of our guards were **adopted from it** |
| `Minimax-H3-Turbo` | `02e26d5` | the vendor README that publishes the distilled sigma grid `bench/check_distill_grid.py` grades against — a grid from the vendor, not one we computed |
| `pdmd` | `03ee66b` | **Retired 2026-10-05 with its lane** (`../roadmap.md`, "Closed lanes"); nothing here reads it any more, and the checkout is the owner's to delete. It was the PDMD release repo (pdmd2026), read 2026-10-01 for the inference scripts that pinned PDMD's sampling contract. By 2026-10-05 its scripts had moved the 2-step audio shift (`AUDIO_SHIFT_2NFE`), which our 2-step probe never followed |
| `ComfyUI-H3-AudioRefine` | `d78d34f` | a third-party pack, added to this table 2026-10-02 (cited before that from `../h3_audio_freeze.md` and `../h3_pdd.md`). It freezes the video stream of a sampled H3 latent and re-denoises only the audio stream through core's per-stream `noise_mask` on the undistilled model, with a K/V cache over the frozen video rows (`coderef/ComfyUI-H3-AudioRefine/README.md`, `TECHNICAL.md`). Read for the audio-refine regime, not installed |
| `MaskVidExperiments` | `c32ed8c` | a third-party pack of video masking tools, added to this table 2026-10-09 and unmoved since 2026-09-11. A mask reduced to the latent grid by the VAE's own frame cycle and H3's token grid, a cleanup of specks, a frame-range mask, an audio time-range mask on a joint latent, a soft variant of Differential Diffusion, and a stable crop round a moving subject that is sampled alone and pasted back (`coderef/MaskVidExperiments/README.md`). Read for the masked lane; not installed. What it does that the lane does not is in "What moved by 2026-10-09" |
| `ComfyUI-NKD-Basic-Tools` | `4f487a5` | a third-party pack, added to this table 2026-10-09. Its crop and stitch pair cuts a masked area out at the model's own resolution, samples it and puts it back, with a colour match on the way (`coderef/ComfyUI-NKD-Basic-Tools/docs/inpaint-crop-stitch.md`). Written for stills; nothing in it is H3-specific. Not installed |
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
| `sam-audio`, `sam-3d-body`, `sapiens2`, `sam3` | Meta's perception models beside the masked lane: sound separation by text, span or mask; a body mesh from one image; human parts and keypoints; the tracker. What each gives and does not is in [`meta_perception_models.md`](meta_perception_models.md) and [`sam3_prompting.md`](sam3_prompting.md) (row added 2026-10-10) | say nothing about H3; none is imported, and a check may run one as a one-off reference |

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
    *2026-09-25: one has been converted.
    `FastVideo/FastVideo-FastH3-Comfy` publishes
    `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`. ComfyUI's own
    templates (`video_fastvideo_fasth3_t2v`/`_i2v`) run it at 8 `simple`
    steps on `res_multistep`, through core's `MiniMaxH3SigmaShift` at 10/3.
    Not on disk here, and no graph of ours loads it.* *2026-09-26: it is on
    disk, and `h3_probe_t2v_fasth3_8step` loads it at the template's settings.
    Those depart from FastVideo's own contract, which this file validates:
    Euler on the release's positions, and VSA sparsity 0.8 (keep 20%) on every
    step, against the template's keep 10% after a dense warm-up.
    `h3_probe_t2v_fasth3_8step_contract` runs the contract
    (`../../bench/results/2026-09-26_fasth3_contract_s1.md`).*
  - `507cb1d83` (#7535) moves its H3 VSA into the model. Same cube tiling and
    gated coarse branch as `vsa_attention.py`; it keeps a fixed count of
    video tiles where ours keeps a fraction (`h3_config.VSA_KEEP_PERCENT`).
  - `5d3e6dc1b` (#7693) rounds the DiT's AdaLN RMSNorm output to bf16 before
    scale and shift, on SM90 only
    (`coderef/vllm-omni/vllm_omni/diffusion/layers/indexed_modulation.py`,
    function `_use_hopper_bf16_affine_semantics` as of `fa506e0fe`).
    Core already applies scale and shift in the input dtype
    (`comfy/ldm/minimax/model.py`, `_mod_scale_shift`).
    *Superseded 2026-09-25: `84977d954` (#7913) deleted that function and
    now keeps the norm and the affine in fp32 on every arch, so the pointer
    no longer resolves. See "What moved by 2026-09-25".*
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

## What moved by 2026-09-25

Read on 2026-09-25 by fetch, from each clone's upstream branch, against the
revision the 2026-09-19 section recorded. The commit list for a clone is
`git log <recorded>..origin/main` inside it (Sana: `origin/sol-engine`).
Diffs were read for the commits named below and titles for the rest. Core,
comfy-kitchen, kijai's fork and ComfyUI's workflow templates live in
[`../sol_upstream.md`](../sol_upstream.md), section "comfy-kitchen and core,
2026-09-25". sglang's seventh read lives in
[`../research/sglang_comparison.md`](../research/sglang_comparison.md).

**Nothing below changes what runs on this card, and nothing triggers the
adopt-upstream rule.** sglang moved no default this repo differs on. The one
upstream default that did move, ComfyUI's templates loading an INT8 video
VAE, belongs to none of the rule's three upstreams and contradicts an owner
decision; `sol_upstream.md` has it. *2026-10-02: no longer a contradiction.
The owner reopened the INT8 VAE on 2026-09-26 and
`workflows/h3_config.py::MODELS` names it as `video_vae`.*

- **`sglang`** (`993d1fccba` to `2f5c9ac43d`) **now implements PDD**
  (`973fb44471`, #40568). It uses the same dt-weighted head fusion as ours
  (`pdd_math.py::fusion_plan`), at `h3_config.PDD_SHIFT` and
  `h3_config.PDD_STEPS` evaluations, over a fixed uniform
  partition fused offline. The seventh read has the comparison, and a probable
  gate/value swap in its offline fc1 merge.
- **`vllm-omni`** (`fa506e0fe` to `3bd5ac968`).
  - **Continuation from the latent tail, as guide rows** (`139a47a57`,
    #7838). The previous window's sampled video and audio tail goes in as
    extra conditioning rows (a `latent_guide` ref block) sharing the new
    window's time origin. The overlap is regenerated and discarded, every
    window's temporal positions are offset onto one global clock, and audio
    boundaries are rounded on the cumulative timeline so no error accumulates.
    Nothing is decoded. `audio_mode=lock_source` pins one encoded track in the
    target audio rows at timestep 1.0 every step, sliced per window, which is
    our mask freeze at `audio_mask` 0 with `track_latent`. **This is the guide
    arm of [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 5 idea 5,
    taken from the latent tail.** Core can express the guide half: its packed
    layout already reads a latent from `minimax_keyframes`. It has no temporal
    offset in its layout, but a pack can add one without editing core (see
    below). The module docstring credits the guide/discard/append algorithm to
    `ttulttul/ComfyUI-Minimax-H3-Continuation`. The recipe credits
    `vizart-vj/ComfyUI-MiniMax-H3-LongMedia` for the global-offset convention
    only. *Corrected 2026-09-25, same day: this bullet first said the
    docstring credited both packs for the algorithm, and that the offset
    needed a core change. All three packs are now cloned and compared in
    [`../research/2026-09-25_continuation_guide_rows.md`](../research/2026-09-25_continuation_guide_rows.md).*
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/continuation.py::diffuse_continuation`,
    `::resolve_continuation`;
    `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/packed_sequence.py`;
    `coderef/vllm-omni/recipes/MiniMaxAI/MiniMax-H3.md`, "Long video and
    driving audio".)
  - **#7693 reversed** (`84977d954`, #7913). The fused RMSNorm and AdaLN
    kernels keep the norm and the affine in fp32 on every arch
    (`coderef/vllm-omni/vllm_omni/diffusion/layers/indexed_modulation.py`).
    sglang still rounds the norm to bf16 before its fp32 affine, as core does
    at the norm, so the removed path was the vendor-compatible one. Core
    rounds at more points than sglang's fused kernel, because
    `comfy/ldm/minimax/model.py::_mod_scale_shift` is two in-place bf16 ops.
    By the vendor reference that is not a defect, and only a capture could
    size it. *Corrected 2026-10-02: sglang's affine is not fp32. Its bf16
    block kernel
    (`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_scale_shift_bf16_kernel`,
    unchanged since 2026-08-18) rounds to bf16 at `1+scale`, at the product
    and at the stored sum, the same three points as core's three in-place
    bf16 ops. sglang and core agree, and vllm-omni's fp32 path is the
    outlier. [`../research/sglang_comparison.md`](../research/sglang_comparison.md),
    "Eighth read".* The same commit turns TF32 off for the keyframe encode on SM90
    only, so [`../open_experiments.md`](../open_experiments.md) #30's
    `allow_tf32=True` quote is now arch-conditional; #30 stays closed.
  - **Steps now count evaluations** (`67aa30c96`, #7219), the convention
    LightX2V and ComfyUI's `steps` already use. Turbo files are validated
    against their declared evaluation count.
    `bench/check_distill_settings.py` reads LightX2V that way already.
    vllm-omni's own client templates still describe sigma-point counts, which
    its server now rejects for the Turbo files it validates. That was read,
    not run.
  - **Exact AdaLN projection cache** (`535bd35b4`, #7987), on by default. It
    reuses the bytes of an identical earlier call, keyed on a digest of the
    whole `t_emb` plus the numeric settings, so hits come across requests with
    the same schedule, not across steps. It is the second engine to do this,
    after LightX2V's `95e9b86b` (2026-09-10 section). Core recomputes.
  - **Latent-mask editing** (`5a93ec1b4`, #7465; client graphs `fa1d03ab8`,
    #7898). Server-side video and audio inpainting whose token pooling and
    mask quantisation match core's
    (`comfy/ldm/minimax/model.py::mask_row_values`), which it cites. Core had
    it first.
  - **Portable prompt skills** (`f8a00b149`, #7923): prompt format only, no
    sampler default. They differ from [`../prompting.md`](../prompting.md) in
    two places: they keep shot-header timestamps, which the house rule forbids,
    and they bind `<Picture 1>`/`<Picture 2>` for FL2VA where the vendor
    guide's alignment line is bare. The vendor guides win.
  - `dbd6a35dd` (#8008) refuses a LoRA whose modules do not all bind; core's
    `comfy/lora.py` still warns and continues. The rest is VAE memory
    (#7241), request cancellation, tests, and first- and last-frame inputs on
    its ComfyUI client pack (#7449), which still runs no H3 weights in ComfyUI.
- **`LightX2V`** (`52161985` to `a4b8ce30`). Under `configs/minimax_h3` there
  are only additions (`git diff --name-status`), so nothing
  `bench/check_distill_settings.py` reads moved.
  - **The "latent cache" is DPCache** (`8652c6f1`, #1557). On unselected steps
    it skips the whole block stack and extrapolates its output with a Taylor
    series. The step set is chosen by a dynamic program over a calibration
    run keyed on the Sol settings. Skipped steps are predictions, not reuse.
    It ships combined with Sol, on ref2av at the base step count, multi-GPU
    (`coderef/LightX2V/configs/minimax_h3/decache/`). **That reverses** the
    rotation survey's E.5 claim that LightX2V refuses feature caching on H3
    and never combines it with Sol; a dated note is in place there. A MagCache
    class landed too but cannot be selected
    (`coderef/LightX2V/lightx2v/models/networks/minimax_h3/model.py::MiniMaxH3Model._init_infer_class`).
  - **Causal streaming RefA2V** (`d43f15f7`, #1539; `40744764`, #1542). A
    second KV-cached causal H3 runtime beside TaoMate, and different on every
    axis:
    - it loads its own full checkpoint, whose provenance the checkout does
      not give;
    - it keeps a rolling K/V cache per denoising step, filled by the noisy
      forward;
    - it caches text through a prefill;
    - it re-ropes cached keys into bounded slots;
    - it puts the driving audio clean in the target rows at timestep 1.0.

    That last point implies a weight set trained with clean audio in the
    target rows, which touches `h3_audio_freeze.md` section 5 item 8. That is
    reasoned from the code. It needs tensor parallelism across several GPUs
    and FlashAttention-3
    (`coderef/LightX2V/lightx2v/models/networks/minimax_h3_causal/streaming.py::MiniMaxH3StreamingPlan`,
    `coderef/LightX2V/lightx2v/common/kvcache/rolling.py::StepRollingKVCachePool`).
- **`ComfyUI-UtilsCollection`** (`1d5b202` to `fc6104c`). Not installed here,
  and no shipped graph wires a `UC_*` node.
  - **A loop sampler** (`UC_H3LoopSampler`, `plan_h3_schedule`) windows one
    whole-clip latent. It carries the previous window's sampled tail through
    `noise_mask`, takes a prompt per window and restarts positions per window.
    That is the same arm as ours, and its decode-and-re-encode path (the
    2026-09-19 note) stays beside it. Its seams are not held to the audio
    grid, where ours require `39+51k`. The reader found several defects in it
    by reading, none run: accepted inputs that are never used, a
    `preserve_input` mode that yields silent audio when no source is wired,
    and an audio mask handed to the video stream as a plain tensor
    (`coderef/ComfyUI-UtilsCollection/helpers/sampling_helpers.py::start_sampling_loop`).
  - `7605f6b` and `e940d37` send standalone reference audio to
    `minimax_refs` only, which matches core's
    `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo` (verified).
    `84c0182` installs object patches on the patcher clone, including its own
    copy of core's H3 `_forward`. It edits no core file.
  - Zero-sentinel inputs (`749d8d6`'s megapixels 0 and others) are noted, not
    port candidates (this pack refuses the pattern:
    `bench/check_literal_widgets.py`).
    Qwen reference collections, fusion images, a prompt builder with a timed
    `Timeline:` block, and a VLM preset that puts ref2va sections into T2VA:
    none contradicts [`../evidence.md`](../evidence.md) "Settled about H3",
    and each departs from the vendor's prompt structure.
- **`flashinfer`** (`dc04f50c` to `bf82326b`): H3 kernels for SM100, SM103 and
  SM120 only. **`Sana`** (`ca26dbd` to `6c2f582`): a HyperFlow doc.
  **`TaoMate-H3`** (`b933d8e` to `6b2f998`): prompt text only.
  **`DiffSynth-Studio`** (`c458cb4` to `7686e54`): Qwen-Image-2.1.
  **`diffusers`** (`a3e0b8ec2` to `bdc2bea37`): no H3 path; `80c7ed262`
  fixes flash and sage varlen prep under `torch.compile`.
  **`Model-Optimizer`** (`b311c054d` to `ed7e87953`): one commit touches the
  `qwen3_vl` recipe files, a recipe-format change read at the title.
- **Unmoved:** `Minimax-H3-Turbo` (`02e26d5`), `MiniMax-H3` (`d21241f`),
  `TurboDiffusion` (`e3d6136`), `TaoMate-LTX` (`136d890`),
  `comfyui_dagthomas`. `alibaba-pai_MiniMax-H3-Acc-LoRAs` is a Hugging Face
  download, not a clone, and its repository's newest commit is 2026-08-27 (HF
  API).
- **Not read:** the infrastructure clones past a commit-message search for
  H3, MiniMax and Qwen3-VL. That found only MiniMax-M3 work in `vllm`.
  Also not read at first: the three third-party continuation packs that
  vllm-omni names (the third is `T8mars/comfyui-minimax-h3-audio-T8`, which
  its recipe cites for a different route). They were cloned and read later
  the same day, in the continuation note linked above. Still not read: the
  line-number citations into the sglang files these commits touched,
  beyond the one corrected in `sol_upstream.md`.

---

## What moved by 2026-10-02

Read on 2026-10-02 by fetch, from each clone's upstream branch, against the
revision the 2026-09-25 section recorded (pdmd and the continuation packs
against their 2026-10-01 and 2026-09-25 reads). The commit list for a clone is
`git log <recorded>..origin/main` inside it (Sana: `origin/sol-engine`).
sglang, vllm-omni, flashinfer and Model-Optimizer were fast-forwarded to
their upstream tips so the paths below resolve; the others already sat there.
Diffs were read for the commits named below and titles for the rest. Core,
comfy-kitchen and ComfyUI's workflow templates live in
[`../sol_upstream.md`](../sol_upstream.md), section "comfy-kitchen and core,
2026-10-02". sglang's eighth read lives in
[`../research/sglang_comparison.md`](../research/sglang_comparison.md).

**Nothing below changes what runs on this card, and nothing triggers the
adopt-upstream rule.** sglang moved no default, and no H3 template moved a
widget value. Two things the read turned up are about our code, not theirs:
an open core PR that would break `MiniMaxH3LoRABranch` (16681, in the
`sol_upstream.md` section), and a claim on this page about sglang's AdaLN
rounding that was wrong when written (corrected in place under 2026-09-25).

A note on the earlier sections: `bench/check_distill_settings.py` has read no
LightX2V config since the turbo lane closed on 2026-09-26 (its docstring, "no
retired turbo"). The sentences in the 2026-09-10 and 2026-09-25 sections that
say it does were true on their dates.

- **`sglang`** (`2f5c9ac43d` to `89f21671bb`). **It now runs the H3 DiT under
  a ComfyUI graph** (`f1e62e3a2e`, #35990): core builds the model and runs
  the encoder, VAEs, conditioning and sampler, and an sglang worker process
  runs each DiT step. None of our DiT-side patches would reach a DiT run that
  way. Also an opt-in Spectrum skip-step, `kitchen_int8` renamed
  `convrot_int8`, an exact conditioning cache and t2va RL rollout. The eighth
  read has each.
- **`vllm-omni`** (`3bd5ac968` to `527982d88`).
  - **Reference stills keep their own size** (`7266fc613`, #8253). An image
    reference used to go to a 2048 short edge, up or down. It now keeps its
    resolution, rounded per axis to 32, inside validation bounds only
    (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/preprocessing.py::resolve_minimax_h3_reference_image_shape`).
    The title says "avoid enlarging", but large stills are no longer shrunk
    either, where core's `max` mode shrinks at its short edge. A reference
    video goes on the canvas only when it has at least the canvas's pixels and
    otherwise keeps its size rounded to 32
    (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/reference_video.py::_reference_video_shape`),
    which is core's rule in
    `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo`. The PR's
    evidence is wall time, not quality. sglang still scales stills to its
    short edge with upscaling on
    (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py`),
    which is what the append node's parity default follows, so that default
    and [`../evidence.md`](../evidence.md) "Settled about H3" stand.
    It is one more data point for the unrendered `allow_upscale` arm in
    `bench/refview2_arms.json`, not a verdict. The PR's before side was read
    from the GitHub diff page, not the clone.
  - **Latent super-resolution and a hi-res refine pass** (`a038b3817`,
    #8322). An optional community 3D-conv latent upscaler
    (`LBH-123-AI/Minimax_h3_latent_Upscaler`, upscale only) runs between the
    denoise loop and the decode
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/latent_upscaler.py::MiniMaxH3LatentResizer3D`).
    `latent_refine` re-noises video and audio to their own shifted sigmas at
    one schedule index and reruns the schedule's tail at the larger size
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py::_refined_latents`).
    Both are off unless a checkpoint path is set. Upscale alone gives a larger
    decode of a canvas-native clip. The refine pass runs the DiT above the
    trained canvas ([`../h3_resolutions.md`](../h3_resolutions.md)). A
    third-party ComfyUI node for the checkpoint exists; it is not cloned or
    read here, and whether core's `LatentUpscaleModelLoader` loads the file
    is unchecked.
  - **FastH3 four-step on Ascend** (`817f5d0e4`, #7149): an offline adapter
    fusion that writes the existing ladder
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3.py::FASTH3_BASE_SCHEDULE`)
    into `model_index.json`. The contract is unchanged; a second hardware
    path now uses the same positions and base shifts.
  - The rest touches no H3 numerics: video transport (`3a82c8588`), an
    offload-flag refactor in which FastH3 still refuses offload
    (`acf662023`), an empty-tensor guard on the shared RMSNorm (`965ad68f1`),
    TTS, audio and other models.
- **`diffusers`** (`bdc2bea37` to `578c9b2c6`).
  - **The H3 VAE decoder now runs in the pipeline dtype** (`51a454be9`,
    #14754). It used to pin every top-level VAE module to fp32 and decode
    under fp16 autocast. Now only the encoder, `quant_conv`, the norms and the
    LayerScale scales stay fp32
    (`coderef/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py::AutoencoderKLMiniMaxH3._keep_in_fp32_modules`).
    That moves diffusers toward core on decode. Core still differs in keeping
    the scales and residual stream in the activation dtype
    (`comfy/ldm/minimax/vae.py::TransformerBlock._residual_scale`); only a
    capture could size that. [`../custom_node_gaps.md`](../custom_node_gaps.md)
    item 3 has a dated note.
  - **Single-file loading for ComfyUI-format H3 files** (`4de185d6e`, #14839).
    Its converter records three namespace facts: Comfy-Org single files stack
    QKV as `[q;k;v]` where the release shards interleave per head, the fused
    FF is `[gate;value]`, and pruned files (with `adaln_t_table` in place of
    `time_embedder`) are refused
    (`coderef/diffusers/src/diffusers/loaders/single_file_utils.py::convert_minimax_h3_transformer_checkpoint_to_diffusers`).
    A third reading of the layout our converters assume.
  - `e0abab83b` downcasts `position_ids` on backends without fp64 only, and
    `fef717ffb` adds an SM120-only Sage backend.
- **`Sana`** (`6c2f582` to `670482d`).
  - **SoL-Refiner's H3 version** is a one-step LTX-2.5 refiner over a
    decoded H3 clip: LTX-2.5's VAE re-encodes it, the latent is upsampled, one
    forward and one Euler step run with no CFG, and LTX-2.5's diffusion
    decoder outputs a higher-resolution clip with no audio. Text goes through
    LTX's own encoder. The weights are a full Diffusers pipeline with merged
    adapters, not a LoRA, tested on an H100
    (`coderef/Sana/models/sol-refiner/MiniMax-H3/sol_refiner_h3/pipeline.py::SoLRefinerH3Pipeline`).
    It uses no Sol on H3. No ComfyUI path is published. Core has LTX-2.5
    parts, but whether it loads this transformer layout is unverified, and a
    24 GB card would need offload or quantization (reasoned).
  - **`1fb0648` (#511)** packages Spark's two-stage recipe for one RTX 5090.
    Its offload runtime refuses anything but SM120
    (`coderef/Sana/models/minimax_h3/Sol-H3-RTX5090/runtime/offload.py`). Its
    Sol on H3 is the opt-in Ref2VA path that
    [`../sol_upstream.md`](../sol_upstream.md) already records for Spark.
    Its checkpoint list loads a lightx2v turbo file, a closed lane here.
  - `1fbe166` makes the LTX-2.3 refiner's attention architecture-aware. The
    shared Sol backend table still maps SM89 to `cute_sm89`.
- **`LightX2V`** (`a4b8ce30` to `8a97c759`). Nothing under
  `configs/minimax_h3` changed.
  - `190f2ef0` (#1497) adds Apple MPS support. Its "Turbo" config is a
    four-step smoke config with no LoRA and no shift
    (`coderef/LightX2V/configs/platforms/mps/`); its VAE fixes are MPS-only
    and a loader key map.
  - `d46ab933` (#1568) adds prompt travel to the causal RefA2V runtime. It
    re-encodes the prompt at chunk boundaries, padded to one fixed token
    length so the text prefix keeps its row count, and rewrites only the
    condition K/V
    (`coderef/LightX2V/lightx2v/models/runners/minimax_h3_causal/action_prompt_travel.py::ActionPromptTravel`).
    `loop_audio` tiles a short driving track. Fixed-length padding is the
    pattern to know if anything here ever swaps prompts per window inside one
    sequence. The runtime still needs its own checkpoint, tensor parallelism
    and FlashAttention-3. `cb61e625` is a one-line config path fix.
- **`ComfyUI-UtilsCollection`** (`fc6104c` to `834d66b`). Not installed here,
  and no shipped graph wires a `UC_*` node.
  - First and last frames become core-style keyframes and also go to Qwen as
    reference pictures (`f767043`). Native image references go to the VAE at
    source size floored to 16, with a separate smaller Qwen view
    (`coderef/ComfyUI-UtilsCollection/helpers/minimax_h3_reference_media_helpers.py`).
    Two different stills for the two towers departs from vendor parity, which
    our `MiniMaxH3AppendRefImage` defaults to
    ([`../h3_references.md`](../h3_references.md)).
  - Its isolated reference-video path (`4f439bd`) does not truncate, snap
    the frame count or apply the canvas rule, all three of which core does,
    and orders labels images, videos, audio, turning a video's soundtrack
    into a standalone audio reference. That departs from the label rules all
    four implementations agreed on in the 2026-08-28 pass. Reasoned, not run.
    **Dated note, 2026-10-05:** the first half no longer holds. Since
    `ed4716e` the path truncates, snaps the frame count and applies the
    canvas rule; the label order still departs ("What moved by
    2026-10-05").
  - "Pooled" and "refined" reference-video modes (`6552e04`) pool the VAE
    latent and then fit it by gradient steps
    (`coderef/ComfyUI-UtilsCollection/helpers/model_helpers.py::_pool_minimax_h3_visual_latent`,
    `::_refine_minimax_h3_visual_latent`). A pooled latent is not the encoding
    of a smaller clip. Its default is still the full, unpooled video.
  - `0251732` sends the continuation tail to Qwen at 2 fps from index 0,
    which now matches core. `4353de3` makes its sage forward's tensors
    contiguous; our `attention.py` splits QKV as core does. Its VLM presets
    keep timed shot headers, the same departure from
    [`../prompting.md`](../prompting.md) as vllm-omni's 2026-09-25 prompt
    skills. One input still treats 0 as "preserve original", which
    `bench/check_literal_widgets.py` refuses here.
- **`DiffSynth-Studio`** (`7686e54` to `974cfa3`). `d393669` (#1712) adds
  EntroPack, pre-quantized H3 DiT, text-encoder and VAE packages at several
  bit widths in its own `CompressedLinear` format, not ComfyUI's
  (`coderef/DiffSynth-Studio/examples/minimax_h3/model_inference/MiniMax-H3-EntroPack.py`).
  The encoder packages touch the closed lane on quantising our own encoder;
  noted, not proposed.
- **`flashinfer`** (`bf82326b` to `11e188412`): H3 kernels for SM100, SM103
  and SM120 only. **`Model-Optimizer`** (`ed7e87953` to `44b46eba8`): IQ
  codecs and other models, titles only. **`transformers`, `vllm`,
  `llm-compressor`, `triton`**: a commit-message search for Qwen3-VL, MiniMax
  and H3 since 2026-09-25 found MiniMax-M3 work and one device-side Qwen3-VL
  normalisation in vllm's own path. Nothing reaches SM89 or our encoder.
- **Unmoved:** `Minimax-H3-Turbo` (`02e26d5`), `MiniMax-H3` (`d21241f`),
  `TurboDiffusion` (`e3d6136`), `pdmd` (`03ee66b`), `FastVideo`
  (`9491c863`), `comfyui_dagthomas`, `ComfyUI-Minimax-H3-Continuation`
  (`e1768d5`), `ComfyUI-MiniMax-H3-LongMedia` (`409e4cb`),
  `comfyui-minimax-h3-audio-T8` (`70fb30f`) and `ComfyUI-H3-AudioRefine`
  (`d78d34f`), now a row in the ComfyUI-side table.

---

## What moved by 2026-10-05

Read on 2026-10-05 by fetch, from each clone's upstream branch, against the
revision the 2026-10-02 section recorded. Every clone but `pytorch` already
sat at its upstream tip, so the commit list for one is
`git log <recorded>..HEAD` inside it. Diffs were read for the commits named
below and titles for the rest. The FastVideo, sglang and UtilsCollection
reads were done by three read-only subagents from saved diffs and the
checkouts; the claims this section rests on about our own code were then
checked by hand (`lora_branch.py::parse_lora`, core's VAE calling kitchen's
fused norm, core's `lcm` against `CONST.noise_scaling`, the open row in
`../SOLATTN.md`). Nothing was run except the Triton probe named below.

**Nothing below changes what runs on this card by itself, and nothing
triggers the adopt-upstream rule: no upstream moved an H3 default.** Three
owner decisions came out of the read, all dated 2026-10-05 in
[`decisions.md`](decisions.md): the base sampler moved to Euler, PDMD was
retired, and `MiniMaxH3LoRABranch` takes Kohya-style keys.

- **`pdmd`** (`03ee66b` to `041b70b`). The trainer's scripts now set the
  audio shift from `AUDIO_SHIFT_2NFE` when `steps == 2`
  (`coderef/pdmd/worker/run_a10.py`, `run_a100.py`; `2fb6cd2`, `ef59058`),
  and keep the old value "for paper metrics". Our 2-step probe never
  followed; the owner retired the lane the same day (`../roadmap.md`,
  "Closed lanes"). The range also adds an `eval/` tree. Not read further.
- **`vllm-omni`** (`527982d88` to `9146284c1`). `b65f97bc4` (#8378) adds a
  request-scoped `res_multistep`
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/sampling.py`).
  Euler stays the default, and a fixed distilled schedule refuses anything
  else. It is what prompted the owner to move our base to Euler:
  `workflows/h3_config.py::SAMPLING` has the provenance.
- **`sglang`** (`89f21671bb` to `efb62ce26`). No H3 sampling default moved:
  the diff of
  `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py`
  adds a step floor constant.
  - **Kohya LoRA keys and a community LoRA table** (`f048d5aa4b`, #35857):
    `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/lora/format_adapter.py` rewrites
    `lora_unet_blocks_N_*` names to native ones, and the cookbook lists
    community LoRAs with strengths and trigger phrases. Core's stock loader
    already maps those names (`comfy/lora.py::model_lora_keys_unet`); our
    at-the-call node refused them until 2026-10-05 and now reads them
    through core's table (`lora_branch.py::native_keys`). None of the listed
    LoRAs is on this box, and no Kohya-trained file has been run here.
  - **The VAE decoder's RMSNorm and QK RoPE fused in Triton**
    (`284cda01fb`, #41906), on unless `MINIMAX_H3_VAE_DECODER_FUSED_NORM=0`.
    Core already does this through kitchen
    (`comfy/ldm/minimax/vae.py`, `ck.rms_rope_split_half_` and
    `linear_input_act(..., "rms_norm")`). Nothing to adopt.
  - **Sparse routing in head chunks** (`1093c501df`, #41985): one call split
    by heads so the routing tensors stay under a cap, output unchanged by
    its own test. Its backend is not SM89. As an idea it would lower peak
    VRAM in our dense-QKV `sol_attn` path on long clips, not speed it up;
    our chunked producer chunks rows, not heads. Noted, not proposed.
  - `150f568900` stops its Triton kernels specialising on sequence length;
    kitchen's Triton backend has no such marking, and ComfyUI leaves that
    backend off (`comfy/quant_ops.py`). `8663e3b670` (no full-video clone
    after decode) and `a977e3b9d5` (stream an oversized DiT) are things
    core already does. `efb62ce269` adds an explicit QKV layout override to
    its loader, which matches the row-order question in
    [`../custom_node_gaps.md`](../custom_node_gaps.md) without closing it.
    `fbce0d9478` lets its own ComfyUI worker read serialized INT8 files.
- **`FastVideo`** (`9491c863` to `6ded84ee`). `8444c089` (#1907) adds
  inference for a FastH3 Ref2VA PDD student, and a reference-video policy
  for VSA: each reference video is its own sparse region, and every video
  query keeps a set share of each reference's tiles and of the target's
  (`coderef/FastVideo/docs/inference/fasth3-distilled.md`,
  `coderef/FastVideo/fastvideo/attention/backends/video_sparse_attn_h3.py`).
  - **Nothing in it runs here.** The contract's tile size has one route, the
    sm_100a/sm_103a kernel, with no Triton fallback (the same doc,
    "Hardware"), and no checkpoint repo id is named in the docs or the
    example.
  - **What carries to SM89 is the idea, and it is already an open row:**
    "Text and audio exact, references sparse, by permutation" in
    [`../SOLATTN.md`](../SOLATTN.md), a permuting `sink_conditioning` mode
    in `sol_attn_h3.py` with no kernel change. Our Sol node keeps reference
    keys in the exact sink today. FastVideo keeps conditioning dense for
    every checkpoint but this student, so upstream treats the policy as
    trained; a training-free version needs a reference capture graded
    first. Separate budgets per reference would need a per-region threshold
    in kitchen's route kernel.
  - Its PDD layer is the algorithm in `pdd_math.py` (fusion of a block of
    heads by shifted-sigma increments, then Euler). Core's
    `BlockSparseAttention` has one `keep_percent` and no reference regions.
    `0cc41a22` is an sm_100a-only fix.
- **`ComfyUI-UtilsCollection`** (`834d66b` to `cdffe30`). Not installed
  here.
  - `ed4716e` makes its reference-video path truncate, snap the frame count
    and apply the canvas rule, as core does
    (`coderef/ComfyUI-UtilsCollection/helpers/minimax_h3_reference_media_helpers.py::prepare_minimax_h3_reference_video`).
    That retires the first half of the 2026-10-02 bullet on that path. The
    label order (images, videos, then all audio) still departs from core.
  - `0602581` adds a "DMAD re-noise" sampler for few-step distills
    (`coderef/ComfyUI-UtilsCollection/helpers/sampling_helpers.py::sample_dmad_renoise`). Its update is
    core's `lcm` on this model type, and it is off the contract of every
    distill we run: FastH3's own scheduler takes an Euler step
    (`coderef/FastVideo/fastvideo/models/schedulers/scheduling_minimax_h3.py`).
  - `8a61115` moves its VLM presets toward
    [`../prompting.md`](../prompting.md) (a shot header only at a real cut,
    three base fields for T2VA); they keep timestamped timelines.
    `10d28c3` adds a resolution and length picker sized by megapixels,
    which leaves the trained canvases
    ([`../h3_resolutions.md`](../h3_resolutions.md)); its length-from-a-clip
    input is the one idea in the range we do not have.
- **`triton`** (main, past `v3.8.0`). Main still pins the legacy `ptxas` for
  architectures below 90 (`coderef/triton/cmake/nvidia-toolchain-version.json`,
  `coderef/triton/third_party/nvidia/backend/compiler.py::get_ptxas`), and `140c33fc3c`
  turns implicit floating-point fusion off by default; no release tag holds
  it yet. The one Triton kernel in our render path, the sage fork's
  `per_thread_int8`, returned bit-identical tensors under Triton's bundled
  `ptxas` and under the system CUDA 13.2 Update 2 one (a probe run
  2026-10-05; the script is in the session's `internal/` notes, and the
  result is not a committed record). When a release carries the fusion
  change, rerun that comparison before trusting sage on it.
- **`pytorch`** is trunk, not what is installed. The installed release is
  whatever `torch.__version__` says; read its code with
  `git show <tag>:<path>` in the clone.
- **No H3-relevant change:** `diffusers` (`578c9b2c6` to `c2798cc78`),
  `LightX2V` (`8a97c759` to `0c2edc12`, one unrelated model),
  `Model-Optimizer` (`44b46eba8` to `1a472379`), `flashinfer` (`11e188412`
  to `188bdd769`, H3 kernels for SM90 and newer only), `transformers`,
  `vllm`, `llm-compressor`: a commit-message search for Qwen3-VL, MiniMax
  and H3 since 2026-10-02.
- **Unmoved:** `Sana` (`670482d`), `DiffSynth-Studio` (`974cfa3`),
  `MiniMax-H3` (`d21241f`), `Minimax-H3-Turbo` (`02e26d5`), `TurboDiffusion`
  (`e3d6136`), `comfyui_dagthomas`, and comfy-kitchen's upstream main
  (`vendor/rebuild_kernel.sh --check`). The continuation and audio packs
  the 2026-10-02 list ends with are not under `coderef/` and were not read.

---

## What moved by 2026-10-09

Read on 2026-10-09 by fetch, against the revisions the 2026-10-05 section
recorded. Every clone sat at its upstream tip but `vllm` and
`Model-Optimizer`, each a commit or two behind, so the list for one is
`git log <recorded>..HEAD` inside it. The sglang, vllm-omni and FastVideo
reads were done by three read-only subagents from saved diffs and the
checkouts; what this section says of them was then spot-checked by hand
against the files it cites, and what it says of core, kitchen and this pack
was read by hand. Core is in this read because the 2026-10-05 one did not
re-read it: its range is `65787d66..08ff3c11` in the ComfyUI checkout.
`bench/run_checks.py` ran against that core with the card masked: one red,
the link check, from sglang's refactor (below), corrected the same day.
Nothing was rendered.

**No upstream moved a default this pack ships, and nothing triggers the
adopt-upstream rule.** Three things are worth acting on, in this order: a
model file or a LoRA can now switch core's own Sol-Attn on with no node in
the graph (core, below); FastVideo published a single-4090 path for FastH3
that is not the contract and not on a public checkpoint (FastVideo, below);
and two ideas in the masking packs that the masked lane has not tried
(the cross-check, below).

- **ComfyUI core** (`65787d66` to `08ff3c11`).
  - **A checkpoint can ask for kitchen's Sol-Attn per layer** (`b26625f2`,
    #16831), and **a LoRA can set or replace that choice** (`f49c531e`,
    #16880, whose message names turbo LoRAs as the use). A
    `comfy_attention.config` entry is now a preference list, and
    `comfy_kitchen_sol` with a `tau` is one of its two methods
    (`comfy/ldm/modules/attention.py::ComfyAttention`,
    `comfy/ldm/modules/attention.py::attention_comfy_kitchen_sol`; core's
    `QUANTIZATION.md`, "Diffusion attention preferences"). The LoRA side is a
    key ending in `.config`, which becomes an object patch
    (`comfy/model_patcher.py::ModelPatcher.add_patches`).
    - **What core's call is, beside ours.** It passes the threshold and the
      scale and leaves every other option at kitchen's default: no exact
      sink for text, reference or audio rows, no dense opening steps, no
      short-call gate. The schema of `MiniMaxH3Sol` (`sol_attn_h3.py`) is the
      list of what ours sets on top.
    - **With one of our attention nodes wired, the file's choice is not
      consulted**, on the calls the node takes and on the calls it declines
      alike: an override is tried first and is handed the stock function
      (`comfy/ldm/modules/attention.py::wrap_attn`). `MiniMaxH3ExactBlocks`
      already sets a preference aside for its own call, whatever function it
      holds (`exact_blocks.py`; its comment still names INT8 only).
    - **With no override, the file decides, silently.** That is every graph
      that runs stock attention, the baseline among them: core's doc says a
      `low_precision_attention` of false does not switch it off. No file
      this pack names carries the key today: a header scan of every
      `.safetensors` name in `workflows/h3_config.py` and in the graphs
      `graph_paths` walks, bench graphs included, found none (a one-off on
      2026-10-09, not a committed check). `MiniMaxH3LoRABranch` refuses a
      LoRA that carries one, by its unknown-key rule
      (`lora_branch.py::parse_lora`); core's own loader applies it.
      A community file or a repacked distill is where one would arrive.
  - **The Linear call was reworked twice and H3's DiT comes out unchanged**
    (`0752bcb2`, #16816; `62c49c4d`, #16861): the two commits leave no net
    diff in `comfy/ldm/minimax/model.py`. The video VAE's fused norm now
    takes the norm module and casts its weight inside `comfy/ops.py`, where
    the 2026-10-02 fix did it at the call. `d91ed5f5` (#16862) makes a
    bypass LoRA's replaced `forward` win over the fused path through a new
    `comfy_force_forward` attribute. `MiniMaxH3LoRABranch` replaces Linear
    forwards as object patches and is not caught by this: the DiT's one
    fused call takes `fc2`'s weight in `MLP.forward`, and the node covers
    `fc2` by replacing the MLP's forward (`lora_branch.py::install`).
    `comfy_force_forward` is core's own switch for that problem, should the
    DiT gain a second fused call.
  - `3d9b2d55` (#16167) adds signed-policy governance of custom nodes. It
    is off in a stock checkout (`app/governance.py`, `GOVERNANCE_REQUIRED`)
    and loads this pack as before. `926d828e` adds LoRA stack loader nodes.
    The pin moved to `0.2.37` (`2472a20b`). No H3 node or template moved a
    widget value: nothing in the range touches
    `comfy_extras/nodes_minimax_h3.py`, and Comfy-Org/workflow_templates
    has no H3 commit since 2026-10-02.
- **comfy-kitchen.** Upstream main is still the commit the installed build
  sits on (`vendor/rebuild_kernel.sh --check`). Open and not carried: #234
  (kijai, draft) tunes INT8 and Sol attention for compute capability 8.0 to
  8.8 and leaves Ada as it is by its own comment, and would conflict with
  what we carry in `comfy_kitchen/backends/cuda/__init__.py`; #224 as
  [`../sol_upstream.md`](../sol_upstream.md) has it under 2026-10-02; #168
  (ours) has not moved.
- **`sglang`** (`efb62ce26` to `f48ed2127e`).
  - **Its FastH3 entry is now the 8-step V2 checkpoint, on the trainer's
    contract** (`4f320fb85a`, #37662). The grid point count, the trained
    rungs and the shifts read from the checkpoint's own metadata, Euler,
    the kept share and the tile are pinned in
    `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`
    and
    `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::FastH3PipelineConfig`,
    t2va only, with no dense opening steps. That is a second engine on the
    values `workflows/h3_config.py` holds as `FASTH3_CONTRACT_POSITIONS`,
    `FASTH3_SHIFT`, `FASTH3_CONTRACT_SAMPLER` and `FASTH3_CONTRACT_VSA`, and
    none of ours moves. Its sparse backend still refuses this card
    (`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::_VideoSparseAttentionH3BackendResolver`);
    the new native kernel in the same commit is for capability 10.0 and
    10.3. Decoding every VAE tile of a rank in one batch is on for FastH3
    only.
  - **The default quality tier changed meaning and kept its name**
    (`7a4d6dfd2f`, #42370). The reference path is `exact` now, and
    `lossless`, still the default, is the tier that allows fused kernels
    (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`).
    For H3 the denoise is the same in both; the difference is the video
    VAE's decode, whose fused path is allowed from `lossless` up
    (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py`,
    the `quality_allows` call).
    A sentence written before this week that calls sglang's default
    "lossless" names the older, stricter tier. The cookbook still calls the
    default by its old alias in places.
  - **VDN-H3 is where it was.** The hybrid attention the owner asked about
    on 2026-10-09 is the 2026-09-19 entry of
    [`../research/sglang_comparison.md`](../research/sglang_comparison.md)
    ("Sixth read"): OpenVDN's own checkpoint, t2va and fl2va, refused on
    base weights. Its files changed this week only in the quality rename
    and a removal of dead helpers. sglang's cookbook lists Ampere and Ada
    as enabled and not benchmarked.
  - `9101e895ea` writes the MP4 while the VAE decodes and decodes audio
    first. `00b21d6415` resets cache state per request. `516fd1c1b5` and
    `33e7b4cc4b` are speed with the bytes unchanged by their own comments.
    `d677848968` (per-sample CFG norm) does not reach H3, which runs one
    positive forward. `aa5551d9b6` and `43b4abd857` are its MXFP8 and Sage
    plumbing. `17b2f35ae7` removes dead helpers; it shortened a file that
    [`../research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md)
    cited by line, which is what turned `bench/check_doc_links.py` red and
    is corrected there. No base-H3 default moved:
    `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams`
    is the same.
- **`vllm-omni`** (`9146284c1` to `4c5541cfc`).
  - **VDN-H3 here too** (`9cc105d1d`, #8439), as `VDNH3_ATTN`: the same
    OpenVDN checkpoint, loaded as two LoRAs fused over the fl2va base plus
    the linear branch, the same two tasks
    (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vdnh3.py::VDN_TASKS`),
    Euler at the checkpoint's own step count and shifts, refused on base
    weights and with any other backend. Its code names OpenVDN's reference
    and not sglang's backend. It has no compute-capability gate; its docs
    record Hopper only.
  - **Masks and edited video**: the cross-check below.
  - `a43cdcdee` (#7989) adds FlashInfer kernels for FastH3's tile, which
    need capability 12, and turns its AdaLN projection offload on by
    default there. `d0b0cfd02` splits the VAE decode into its own stage.
    `06afa310c` and `88a35c092` are reference-video speed. No default moved.
- **`FastVideo`** (`6ded84ee` to `2b164405`). **No value of the FastH3 V2
  contract moved**; the contract files are untouched in the range.
  - **A single 4090 is measured, and not on the contract as shipped**
    (`8b5138046`, #1919). The record is
    `coderef/FastVideo/scripts/benchmarks/minimax_h3_4090/README.md`: a
    private pruned FP8 checkpoint, the contract's kept share and tile, the
    DiT streamed layer by layer, and in its headline rows an INT8 Q/K
    kernel its own source calls experimental
    (`coderef/FastVideo/fastvideo/envs.py::FASTVIDEO_H3_VSA_SM89_KERNEL`,
    whose default is the kernel that was already there). The README's
    announcement that V2 runs on a 4090 has no recipe and no table for the
    public checkpoint. V2's tile has had a Triton route throughout; the
    route with no fallback in the 2026-10-05 section is the Ref2VA
    student's tile, and still is.
  - **"FastH3 Trim" is a separate pruned checkpoint** (fewer blocks, a
    low-rank AdaLN), on V2's schedule, with recipes for DGX Spark and MLX
    and none for an RTX card (`e235b6a33`, `02a027c49`). "CompactH3" is a
    third thing: a dense prune on a shorter schedule, for the 5090. Neither is a
    profile of V2.
  - **FastVideo now refuses base H3 under its sparse backend**
    (`coderef/FastVideo/fastvideo/pipelines/basic/minimax_h3/vsa_guard.py::refuse_zero_initialized_h3_vsa`):
    gates that are all zero raise. Training-free sparse attention on the
    base model is not something the trainer's own code runs any more.
  - **The Ref2VA student has a named repo**, in a benchmark README only
    (`coderef/FastVideo/scripts/benchmarks/fasth3_omniref_b200/README.md`).
    Its tile is still Blackwell-only. `e1e255935`'s default-off switches
    are data movement and rounding-matched kernels, measured identical on
    B200.
  - **Ref2VA training shows references the way inference does**
    (`d5287f835`, #1757; `coderef/FastVideo/fastvideo/pipelines/basic/minimax_h3/reference.py`):
    stills at `MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE`, a reference video at
    the canvas and at the model's frame rate, the text encoder's copy of it
    at `MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS`, visual anchors noised once at the
    keyframe level, reference audio clean. It agrees with
    [`../h3_references.md`](../h3_references.md) and with the level
    `reference_noise.py` defaults to. A LoRA trained there is exported
    merged; a standalone H3 adapter is not supported.
  - `8348e83e8` imports Comfy-Org's NVFP4 text encoder file and, below
    Blackwell, dequantizes it per layer. `a8688dddd` is its own LoRA
    unmerge bookkeeping. FP8 on H3 now covers the MLP as well as the
    attention projections. The rest is Blackwell, Hopper, Spark and
    multi-GPU.
- **`ComfyUI-UtilsCollection`** (`cdffe30` to `6fb9163`). Not installed.
  `f1576a6` resamples a video input to the model's frame rate by picking
  frames and returns its audio prepared for H3; `4b81485` keeps every
  reference image when a media config is wired; `aad325d` moves the
  motion-worded VLM presets into their own node; three commits build a
  landmark-warped face composite in pixel space, which no lane here uses.
- **`LightX2V`** (`0c2edc12` to `b6d38283`). `b6d38283` counts only
  standalone audio references against its cap, where a reference video's
  soundtrack used to count as well. `913a974d` adds a contiguous offload
  layout and an optional fp32 LoRA merge. Nothing
  `bench/check_distill_settings.py` reads moved.
- **`Model-Optimizer`** (`1a472379` to `5a7bb6764e`). `9ceca982a6` adds a
  Parallel Decoding Distillation trainer, with a Qwen-Image example and no
  H3 one: the first public PDD training code in a checkout here. The PDD
  lanes are closed; it is a reference for the method.
- **No H3-relevant change:** `DiffSynth-Studio` (`974cfa3` to `acf2ad2`,
  Qwen-Image), `diffusers` (`c2798cc78` to `1d5d056ec`), `flashinfer`
  (`188bdd769` to `50829ac26`, H3 kernels for capability 10 and 12 only),
  `transformers`, `vllm`, `llm-compressor`: a commit-message search since
  2026-10-05. `triton` main still has no release tag holding `140c33fc3c`.
- **Unmoved:** `Sana`, `MiniMax-H3`, `Minimax-H3-Turbo`,
  `comfyui_dagthomas`, `MaskVidExperiments`,
  `ComfyUI-H3-Motion-Context-MultiRef`, `workflow_templates`' H3 files.

### Masks and edited video upstream, beside the masked lane

The owner's ask, 2026-10-09. Sources: vllm-omni's latent-mask editing
(server from `5a93ec1b4`, recorded 2026-09-25; this week's `25fb25766`,
#7947, and `bad88bca2`, #7575, add frame-space masks and thin its ComfyUI
client), core, `MaskVidExperiments`, and FastVideo's reference preparation.
sglang's own pipeline has no mask path; the 2026-10-08 read is
[`../research/sglang_comparison.md`](../research/sglang_comparison.md),
"Ninth read". DiffSynth, diffusers, LightX2V and FastVideo have no H3 mask
or edit path (a search of their H3 code for the words, 2026-10-09).
The lane's own map is [`masked_v2v.md`](masked_v2v.md).

**Where the lane and upstream agree**

- **What the model is shown in the kept region.** Core mixes the source's
  clean latent with noise at the reference level and labels those rows at
  that level, on every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`,
  `VISUAL_COND_TIMESTEP`), and writes the clean latent back after.
  vllm-omni does the same and says it matches core
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/latent_mask.py::_quantize`,
  its docstring).
  The lane samples through core, so it is the same rule. The level is the
  constant, not the payload's: `MiniMaxH3ReferenceNoise` moves how clean
  the references are shown and does not reach the kept plate.
- **Which frames a latent covers.** vllm-omni pools a per-frame mask over
  the same frame cycle (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/encoder_processing.py::_temporal_group_max_pool`)
  that `video_mask.py::run_lengths` takes from core's `FRAME_PER_TOKEN`,
  with the maximum, as `video_mask.py::token_mask` does: one regenerated
  frame regenerates its whole latent. `MaskVidExperiments`' Mask To Latent
  Space is the same reduction, and its README shows, on LTX, what
  ComfyUI's default resize does to a mask that skips it.

**Where they differ, and what each difference is**

- **A token is kept whole here; upstream keeps cells inside a token.**
  `token_mask` thresholds, then makes every cell of a token alike. Core
  and vllm-omni pool to the token for the model's label and keep the mask
  per latent cell for the restore, so half a token can be the source.
  vllm-omni's new frame-space intake also resizes by area, so an edge cell
  holds a fraction and its token is labelled part-way
  (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/encoder_processing.py::_resize_video_edit_mask`).
  The lane's choice is deliberate (a token regenerated or kept whole), and
  what a fractional label does was reasoned on 2026-10-05
  ([`../research/masking/2026-10-05_mryellow.md`](../research/masking/2026-10-05_mryellow.md),
  section 1). Nothing here has rendered a cell-level edge beside a
  token-level one.
- **The task and the references.** vllm-omni's shipped editing graph runs
  t2va on the fl2va weights with no reference at all: the mask and the
  prompt are the whole instruction. The lane runs ref2va with a still, and
  on the motion graphs the subject's own frames as a reference video.
  Their server does not refuse references with a mask; no graph or test of
  theirs that was read sends both.
- **The source's shape.** vllm-omni scales the source straight to the
  canvas, so a source of another aspect is stretched
  (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/reference_video.py::prepare_edit_video`).
  `video_mask.py::fit_frames` centre-crops, as core's own nodes do.
- **The margin.** Their graph's note asks for a margin of one token's
  width round the object, as advice. The lane's is `grow_pixels` or a
  share of the subject's size (`video_mask.py::margins`).
- **Audio.** A fractional audio mask in some of their graph's cases, which
  lets the track move part of the way; a time-range audio mask in
  `MaskVidExperiments`. The lane freezes the whole track
  ([`../h3_audio_freeze.md`](../h3_audio_freeze.md)).
- **Specks.** `MaskVidExperiments`' cleanup labels blobs across time, so a
  blob that is small and brief goes and a small one that persists stays.
  `subject_tracks.py::drop_specks` judges each frame alone, by size and
  distance from the largest piece.

**What upstream does that the lane has not tried**

- **A crop round the subject, sampled alone and pasted back**
  (`MaskVidExperiments`' Subject Crop and Uncrop; NKD's crop and stitch
  for stills). The crop is planned over the whole clip so it holds still
  under mask jitter and moves only with sustained motion, and it can be
  enlarged to the model's resolution. It is upstream's answer to a subject
  who is small in the frame: the subject gets the canvas's tokens, where
  the lane gives a small subject a small share of them and has been
  shrinking the region instead. What it costs on H3 is unknown: the plate
  inside the crop is an enlargement, the room outside it is not seen, and
  a crop that moves is camera motion to the model. Untried here, and not
  proposed as a build before the owner has seen it on one clip.
- **Keeping a stretch of time and regenerating the rest.** vllm-omni's
  temporal mask node builds a per-frame mask for continuation and for
  extension, with the kept length snapped to the model's frame rule;
  `MaskVidExperiments` has a frame-range mask. `bench/patch_render_window.py`
  is the lane's use of the same axis, for a hole in the middle.
- **A soft edge during sampling.** `MaskVidExperiments`' soft variant of
  Differential Diffusion keeps a feathered edge feathered at every step.
  The lane's edge is hard at the token and softened only in the pixel
  composite. Whether core's per-token label and a per-step threshold
  compose on H3 is not known.

Three questions the whole-frame lane asked the same day (the first sigmas
of a late start, the last latent frame of a window, one fraction as the
mask over the whole frame) are answered from code in
[`../research/masking/2026-10-09_mrfetch.md`](../research/masking/2026-10-09_mrfetch.md).

---

## The streaming references: TaoMate

Read 2026-09-15, at the revisions named here. The lane that used them is
deprecated, not pursued (2026-09-27); the checkouts stay as references.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `TaoMate-H3` | `ccc1a70` | TaoLiveAIGC's streaming runtime for H3. A 3-step LoRA on the FL2VA partition, run over each 5-second request in causal chunks: the chunk's video attends to the prompt, to a clean K/V cache and to itself, and a sigma-zero forward after each chunk commits its K/V. The cache keeps the first chunk's video as a sink and the two most recent chunks (`coderef/TaoMate-H3/src/taomate_h3/streaming/cache.py::CleanAVKVCache.retain_sink_and_recent_commits`). It accepts only 4 or 8 GPUs under TP2 with Ulysses and requires FlashAttention-3 (`coderef/TaoMate-H3/src/taomate_h3/config.py::DirectRunConfig`, `coderef/TaoMate-H3/src/taomate_h3/streaming/attention_hook.py`), so it does not run on this box. This pack's port of it (a converted LoRA, probe graphs and a streaming sampler node) was deprecated by the owner on 2026-09-27 and removed ([`decisions.md`](decisions.md)); its records stay in `bench/results/` (`2026-09-15_taomate_*`) | you need the adapter's distilled sigma grid (`coderef/TaoMate-H3/src/taomate_h3/model/pipeline.py`, `DISTILLED_STATE_INDICES` at its two shifts), how a KV-cached causal H3 is wired, or an H3 team's own audio-freeze regime |
| `TaoMate-LTX` | `136d890` | the same group's LTX 2.3 system and the code for their paper (arXiv 2607.24359): learned persistent memory, reference-aware FiLM, a pyramid K/V retention policy, stage-parallel inference | the paper's mechanisms. Not evidence about H3 |

What the H3 checkout is not evidence of:

- **The paper's memory.** The H3 adapter holds LoRA factors on the existing
  attention and MLP linears and nothing else (the removed converter refused
  any other tensor; its record is
  `bench/results/2026-09-15_taomate_lora_conversion.json`). The H3 runtime's only
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
  outside anything its authors run. This is the audio-freeze regime
  ([`../h3_audio_freeze.md`](../h3_audio_freeze.md)).
- **Anything about a single-card graph.** The README's timings are its own
  base runtime on its own node.

The community ComfyUI copy: kijai's `minimax_h3_taomate_3step_lora_avg_rank_19_bf16`
is a per-module truncated SVD of this adapter, in the same qkv and SwiGLU
layout. How much of each delta it keeps is in the conversion record's
`comparison`, measured against the full-rank conversion.

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
