# Upstream synthesis, 2026-11-15: what is out there, what it could do for us, how deep we looked

A dated research record. It gathers everything found on 2026-11-15 across the upstream engines, ComfyUI core and the
periphery, and groups it by area of focus, with what each item could do for the projects in this repo. Every item
carries a depth tag, so the reader can tell code that was read from a PR title that was glanced at.

**Where it came from.**
- The 2026-11-15 sweep: ranges from the 2026-11-04 tips to that day's tips, read by five read-only subagents and
  `bench/upstream_watch.py` ([`../upstream/sweeps/2026-11-15.md`](../upstream/sweeps/2026-11-15.md)).
- A second pass the same day over sglang's and vllm-omni's open H3 PRs.
- Later the same day, at the owner's question, a pass on SAM 3.1 (Meta's repo, core and MAINodes) and a dated
  provenance ledger between this pack and `matlowai/ComfyUI-MAINodes` (sections 15 and 16), and an outcome-first
  comparison of what each side could take from the other (section 17), corrected after a Gemini review. The ledger was read from
  both git histories by a read-only subagent, with its load-bearing commits spot-checked by hand.
- The lead session's own hand checks.

The standing pages that keep this current are in [`../upstream/`](../upstream/index.md): engines, periphery, topics
and the open-work horizon. This page is a snapshot of 2026-11-15 and is not updated after that day.

**Trust**, the owner's ranking set during this sweep ([`../upstream/index.md`](../upstream/index.md), "When upstreams
disagree"):
- sglang, then vllm-omni, then diffusers;
- ComfyUI core trusted least as evidence, but watched most because the pack runs on it;
- everything else is periphery, read for how it differs, aligns or does something new;
- community custom nodes are community research.

Figures quoted below are their authors' claims unless tagged H.

**Nothing in the pack was changed by this work.** One related change landed from another session the same day; see
"References" below.

### Depth tags

| tag | meaning |
|---|---|
| **H** | checked by hand by the lead session: the code, the file on disk or the PR body was opened and read |
| **C** | code or diff read by a sweep subagent; not re-checked by hand |
| **B** | a PR or issue body, README, model card or abstract read (by an agent unless marked H); no code |
| **T** | title or listing only |

---

## 1. Masked video-to-video and character swaps (the masked lane, `docs/wiki/masked_v2v.md`)

- **Viggle Remake** (Hugging Face `Viggle/Viggle-Remake`; card **B**, file listing **H**).
  - Two rank-256 LoRAs, `swap` and `edit`, on the base transformer.
  - Inputs: one to four painted `<Picture N>` frames and a required six-section prompt. Windows fall inside 90–345
    frames.
  - It ships `PROMPTING.md`, `COMFYUI.md` and an agent path.
  - *For us:* a second public method for the lane's exact task, with a prompt guide to read against ours.
- **vllm-omni character-replacement skill** (`b8f128171`, #9194; **C**).
  - Reference-only conditioning, no pixel lock, the Character Swap LoRA, six-section `<Subject N>` prompts, videos
    resampled to 24 fps first.
  - Two unmeasured claims: segments of about five seconds hold identity better than longer ones, and a segment can
    be generated backward (reverse the source, render, flip the result).
  - *For us:* segment length and the backward trick are cheap to try on a hard load.
- **Community swap chains** (**B**).
  - `TheBeast-Spec/minimax-h3-multi-swap-comfyui`: SAM 3 tracking, ref2va with the swap LoRA and DWPose, then a
    BiRefNet paste-back, chained over several people.
  - `wwzhifeng/ComfyUI-WZF-VideoRemake`: a one-click Viggle wrapper with cuts, keyframe painting and prompt writing.
  - `chemny/minimax-h3-dance-replica`: an agent skill.
  - *For us:* how others order perception, generation and paste-back.
- **Masks on AV latents, the community's answers** (**B**). `johiny/comfyui-h3-masked-latent` wraps a latent as
  `NestedTensor(video, zero audio)` so stock `SetLatentNoiseMask` works. `estebencalcina/ComfyUI-MiniMaxH3-AVMaskBridge`
  strips or merges clashing masks. *For us:* confirms the problem `MiniMaxH3MaskedSource` solves; nothing to adopt.
- **Removal LoRA** (`jialidddh/MiniMax-H3-People-Vehicle-Removal-LoRA`; **B**): a ref2va removal LoRA run in DiffSynth.
  *For us:* a possible stranger and passer-by cleanup tool.
- **MAINodes** (**C**): a still-region edit kept only inside a SAM 3.1 mask of named words, multiplied into the
  paste-back feather; an eye-colour shift toward a reference through two SAM texts. *For us:* small ideas for
  accessory and eye restores.
- **vllm-omni multiframe reference workflow (#9420, #7558) and Fun ControlNet Union workflow (#7507)** (**T**). Not read.

## 2. Long clips, chained song loops and audio (`audio_freeze_song.py`, `docs/wiki/core/h3_audio_freeze.md`)

- **Latent carry against pixel round-trips** (`matlowai/h3-long-video-research`; **B**).
  - Carrying the sampler latent between windows avoids the colour creep that a decode and re-encode causes.
  - A plain carry eventually grows a mosaic texture, which periodic re-anchoring cures.
  - The mirror case: WanGP #2631 (**B**) reports ref2va continuation turning waxy from the third segment, because
    each start frame is already degraded.
  - *For us:* the most relevant finding for chained windows. Worth reading against the song-loop handoff.
- **vllm-omni sliding windows past 15 s** (#6779; body **H**).
  - Each continuation is a first-frame request from the decoded handoff frame, and the overlap is resolved on the
    latent grid.
  - The previous window's tail audio rides along as a frozen audio reference block.
  - *For us:* the audio handoff is an idea for window joins. The video handoff is the pattern the community notes
    blame for colour creep.
- **vllm-omni app long-clip defaults** (**H**). Window 311, step 17, overlap 39
  (`coderef/vllm-omni/apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/nodes.py`). 311 is off `h3_rules.on_both_clocks`.
  *For us:* a point where their chain rule differs from ours. Our rule stands.
- **Other long-clip packs** (**B**). `KadoBOT/ComfyUI-H3-Continuity-Studio` (visual memory across shots on core
  context windows), `igorls/comfyui-h3-editor` (an editor-style timeline), `JayDataEngineer/ComfyUI-MiniMaxH3-Context-Loop`
  (vocals drive lip sync while the full mix is the soundtrack), `Yuishi1/ComfyUI-MiniMaxH3-lanxi`. *For us:* the
  Context-Loop vocal split touches `MiniMaxH3FreezeAudio`'s choice of what to freeze.
- **`H3RetimeAudio`** (MAINodes; **C**): a soundtrack for a dilated timeline, with short equal-power crossfades and
  transient anchors.
- **Audio VAE numerics.**
  - vllm-omni #9777 (**C**): VAE weights and BigVGAN kaiser-window buffers built while bf16 is the default dtype are
    rounded for good. Not checked against core.
  - vllm-omni #8813 (body **H**): the audio VAE's transposed convolutions rewritten as phase-decomposed conv1d, for
    Ascend.
  - diffusers #15199 / #15245 (**C**): ref2va soundtracks not at 32 kHz.
  - *For us:* the kaiser-buffer question is a cheap check on core's audio VAE load.
- **Circular latent upscale for loops** (`comfyrob/ComfyUI-MiniMaxH3-CircularUpscale`; **B**).

## 3. References and ref2va (`reference_conditioning.py`, `docs/wiki/core/h3_references.md`)

- **Reference-still sizing.**
  - sglang moved its native H3 video default from enlarging small stills to a 2048 short edge to not enlarging
    (`92a6910ef2`, #47907; code and PR body **H**). It was a speed PR, and its own quality table is slightly below the
    old mode.
  - vllm-omni keeps each still's own size (**H**). Core never enlarges (**H**). diffusers, DiffSynth and LightX2V
    still enlarge (**H**).
  - The full comparison and the case each way are on [`../upstream/topics/reference_sizing.md`](../upstream/topics/reference_sizing.md).
  - **State in the pack:** on 2026-11-15 the owner first held the default (`docs/wiki/decisions.md`). Later that day
    commit `2ca5f2d0`, from another session and citing an owner decision, set `MiniMaxH3AppendRefImage.allow_upscale`
    to `False` (`reference_conditioning.py`, **H**). The topic page now records it as decided.
- **Configurable short edge** (sglang #35803; body **H**). Its table shows reference tokens growing with the square of
  the edge. *For us:* a middle ground between on and off, such as a smaller edge for head crops.
- **Video reference only for the first steps** (vllm-omni #9762; **C**, body). It conditions on the reference for the
  first 5 of 20 steps, then drops it from the layout, and reports a match with baseline (NPU). *For us:* independent
  support for `MiniMaxH3ReleaseReferences`, which already drops released rows from the layout
  (`control_vector_nodes.py::_drop_cond_from_layout`). *Corrected the same day: this line said dropping rather than
  releasing would be new.*
- **Reference isolation** (vllm-omni #9689; **C**): reference queries read only reference K/V, captured once. Not
  lossless, by its author. **Reference-K/V refresh around cache hits** (#9728; body **H**). *For us:* both are in the
  family of `MiniMaxH3FrozenVideoCache`.
- **Per-image reference and keyframe row cache** (sglang `b7ccc7e2f9`; **C**): the same idea as
  `MiniMaxH3EncodeReferences`.
- **Pika in sglang inherits the new sizing**, while Pika's own `buckets.yaml` enlarges (agent's reasoning, **C**).

## 4. MiniMax-H3-Image and P-H3-Image (the image lane, the studio app, `h3_image_nodes.py`, `ph3_image_nodes.py`)

- **The Euler step form** (**H**).
  - sglang (`.../minimax_h3/denoise_loop.py`, `minimax_h3_image_denoise_loop`) and diffusers (`euler_update_form="velocity"`,
    `875373ac7`) both step H3-Image as `x + (sigma - sigma_next) * v` in float32. Our graphs use core's `euler`.
  - diffusers claims the heads were distilled with the velocity form and that the difference shows (claim not tested).
  - *For us:* a one-seed A/B would settle whether it matters
    ([`../upstream/topics/h3_image_sampling_step.md`](../upstream/topics/h3_image_sampling_step.md)).
- **Sigma rounding and reference noise order** (**C**). diffusers and DiffSynth derive sigmas as `1 - (1 - sigma)` in
  float32, which moves the sigmas below 0.5 by one ulp. The three implementations draw reference noise in different
  orders. *For us:* only seed-exact parity.
- **AdaLN time-embedding order** (DiffSynth and diffusers **C**; core's cast **H**). They run SiLU on the fp32
  embedding and cast after; core's non-curve branch casts first. *For us:* reaches only `H3_IMAGE_UNET_BF16`, not the
  pruned file the graphs load (reasoned).
- **Who else runs H3-Image** (**C**):
  - core's own path (`ede4e5fe8`);
  - DiffSynth, with a loader for Comfy-Org's pruned int8 file (`28638042`);
  - LightX2V (`0f66206a`);
  - the Comfy-Org templates, on stock `KSampler` euler/simple with a `TextGenerateV2` expander;
  - a vllm-omni single-4090 recipe with no figures (`1b51ce9c4`).

  *For us:* several references to diff our image graphs against.
- **Prompt enhancement** (diffusers docs, **B**): Hub blocks run `MiniMaxAI/MiniMax-H3-Image-PE` and return `prompt`
  and `aspect_ratio`. The code was not read. *For us:* compare with `MiniMaxH3ImagePromptExpander`.
- **Prompt-embedding cache for H3-Image** (vllm-omni #10369; **C**). **Prebuilt AdaLN cache** (sglang #50247; **T**).
- **Community image tricks** (**B**):
  - `Axiris/ComfyUI-AXIRIS-H3-Image-Soft-Start`: a black picture active for the first two of six steps, claimed to
    soften over-contrast.
  - `tritant/ComfyUI_Minimax_H3_Image_Conditioning_Plus`: neutral vision patches for text-to-image.
  - `benjiyaya/ComfyUI-H3Image-LayoutKit`: a bbox canvas turned into a layout prompt.
  - `DeepNeuralNerd/H3-Video-LoRA-Loader-for-Image`: video LoRAs on H3-Image by dropping the audio and final-layer
    keys.
  - `akhaliq/MiniMax-H3-Image-Character-Swap`: the video swap LoRA merged into H3-Image.
  - `an80sPWNstar/comfyui-h3-image-batch`: a batch workaround.

  *For us:* soft-start costs one render to try. The video-LoRA loader opens the video LoRA catalogue to the image
  lane.
- **Core issue #17692 / PR #17705** (**C**): H3-Image checkpoints in the full time-embedder layout reported as pure
  noise. `bench/convert_h3_image_dit.py` writes that layout, and our converted file rendered correctly
  (`bench/results/2026-11-11_h3_image_weights.md`).
- **A tiny VAE for H3-Image** (KJNodes `ff6bdf6c`; **C**): previews.

## 5. Pika H3 Flex (the default distill, `workflows/h3_config.py` `PIKA_*`)

- **Three engines agree on Pika's RNG streams**: initial noise `seed`, re-noise `seed + 1`, condition noise
  `seed + 7919`. They are sglang `f98e9611bc` (**C**), LightX2V `1053bb0a` (**C**) and vllm-omni draft #9777 (**C**).
  *For us:* the target if seed-exact parity with Pika's release is ever wanted. Our draw order was not compared.
- **sglang's Pika numerics** (**H**): TF32 matmul off, and the release's fp32 LoRA factors cast to bf16, unmerged.
  `MiniMaxH3LoRABranch` casts its factors to the compute dtype too (**H**).
- **vllm-omni #9777** (**C**): an option to keep the official bf16 rounding points, claiming near-identity on t2va and
  fl2va.
- **LightX2V six-step configs** (**C**): the same knots, shifts and no-CFG as ours.
- **Pika's own repo** (**C**): `6cb647e` keeps 51 of 64 encoder layers, which does nothing for us since core
  truncates to 50. Issue #9 asks how the distill was trained; it is unanswered.
- **Periphery** (**B**): `sepiablue-ai/ComfyUI-H3-Pika-Flex-Fast` (Pika on core `BlockSparseAttention` with the Sol-H3
  int8 VAE decode, a speed claim), `friendsforhumanity/Pika-H3-Flex-FRef2VA` (first and last frame pinned alongside
  ref2va references), FastVideo's NVFP4 Pika (Blackwell), and vllm-omni #10307 (a request for Pika LoRA loading).
- Standing page: [`../upstream/topics/pika_parity.md`](../upstream/topics/pika_parity.md).

## 6. The Qwen3-VL-32B encoder (shared by every model the pack runs)

- **Agreement** (**C**): layer-50 state before the final norm, bf16, no chat template, the same `Picture i: `
  presentation for H3-Image across engines, and no prompt rewriting in any serving path.
- **sglang rebuilds the vision RoPE table on the GPU for image releases only** (`6be0d75ae0`, #49574; **H**). A CPU
  build computes `pow` differently. *For us:* a possible source of image-versus-video or run-to-run drift. Where core
  builds that table is unchecked.
- **vllm-omni `text_encoder_cudnn_sdp`** (**H**): cuDNN SDPA in the encoder makes the same seed give different images
  across restarts. *For us:* the same question for core, unchecked.
- **Core changes under the encoder** (**C**): `CombinedLinear` fusion (`48b090cb6`), with `MLP.forward` changed but
  unpinned, and video input for Qwen 3.x. `bench/check_reference_encode.py` passes at core HEAD (**H**, run).
- **Quantized and modified encoders** (**B**): an INT4 ConvRot 32B in the H3-Image 50-layer spec (`Merserk/...`);
  abliterated 32B encoders (title or front matter only). *For us:* the INT4 one could matter for 4090 memory if
  quality holds. Untested.
- Standing page: [`../upstream/topics/qwen3vl_encoder.md`](../upstream/topics/qwen3vl_encoder.md).

## 7. Steering, control vectors and prompting (`control_vector_nodes.py`, `j_lens.py`, `edit_guidance.py`, `docs/wiki/prompting/`)

- **KJNodes H3 prompt weighting** (`26e6d342`, `57f4111e`; diffs **C**, commit listing **H**): a per-token attention
  bias where `(word:-1)` removes a concept, plus `[a|b:-2]` blending. *For us:* a different lever from post-refiner
  vectors. Worth comparing on a concept-removal case.
- **Concept Sliders** (`quinteroac/ComfyUI-MiniMax-H3-Concept-Sliders`; **B**): concept text injected *before* the
  token refiner as a model patch, where ours act after it. *For us:* a direct comparison point for `MiniMaxH3Steer`.
- **MAINodes dials on the image model** (**C**): the same text seam `MiniMaxH3Steer` uses. Upstream says the image
  model needed several times the video strength.
- **What MAINodes and this pack took from each other, and their follow-up tests on our ideas** (sink token, span
  pooling at matched size, image-to-video strength, the family check, word-dial centering): section 16.
- **Prompt guidance elsewhere.** vllm-omni's skills defer to MiniMax's `h3-prompt-writing` skill (**C**). LightX2V and
  community rewriters exist (**B**). The paper VideoGen Model 1 (2610.29407; abstract **B**) compares prompt expansion
  against formatting on a post-trained H3.

## 8. Attention, caching and speed (`sol_attn_h3.py`, `docs/wiki/attention/`)

- **The token refiner, once or every step.**
  - vllm-omni #5899 (draft; body **H**) runs it once per request, since it has no timestep dependency, and claims
    bit-identical output.
  - Core #17715 (open; **C**) would run it every step.
  - *For us:* the pack's assumption of once before step 0 (`lora_branch.py`, `nodes.py`) has an engine on its side.
    If core merges #17715, re-read.
- **Attention chosen by sigma window** (vllm-omni `62876f40a`, `e25c9682a`; **C**). *For us:* Sol's dense-or-sparse
  choice keyed on noise rather than step index.
- **Step caching for H3** (bodies **H**): TeaCache for ref2va (vllm-omni #5952) and AlignCache (#9222). *For us:* a
  speed lane on base ref2va. The owner runs no fast distill except Pika, and caching is a different trade.
- **The frontier pushed** (vllm-omni #9959; body **H**): a ref2va latency ceiling (device-resident reference K/V,
  graded sparsity in the dense head and tail windows) that its author says fails the quality bar. *For us:* a map of
  where speed breaks quality.
- **sglang**: `quality="high"` as a fixed Cache-DiT step mask (**C**); a fused gated residual and norm2 (**C**). The
  cookbook says modes differing only in in-op rounding diverge over a long schedule (**C**), so same-seed
  cross-engine A/Bs are read with that in mind. Roadmap #43313 names RTX 4090 defaults and SM89 FP8/W8A8 (**C**).
- **Mixed resolution** (**B**): MPEG-Diffusion (`johndpope/MiniMax-H3-MPEG-Diffusion`, `scrya-com/mpeg-diffusion`)
  and the release's draft PR #91, Level-of-Token, which its author says falls below dense quality.
- **Not for one 4090** (**T** / **C**): Ulysses, tensor and sequence parallelism, disaggregated serving, DGX Spark,
  NPU, XPU and ROCm work.

## 9. VAEs and decode

- **Sol-H3-VAE** (`Efficient-Large-Model/Sol-H3-VAE`; listing **H**; WanGP #2625 **B**): a non-causal int8 video
  decoder reported faster and sharper than LightVAE. *For us:* the pack has no reference to it. A decode A/B on an
  existing latent would be cheap.
- **Tiling artifacts and bit drift** (**B** / **C**): stable-diffusion.cpp #2310 (tiles over 256 px leave a 16 px
  block grid); vllm-omni #10256 (batching tiles changes fp16 bits); diffusers issue #15119 (full-clip fp32
  temporaries after decode).
- **FP8 decoders** (bodies **H**): sglang #45180 needs SM100+, so it is not usable on the 4090. vllm-omni #6131
  applies online FP8 to the decoder's linears; its hardware requirement was not read.
- **Faster save and preview** (bodies **H**): sglang #49080 (NVENC MP4 encode) and #37998 (`taeh3` tiny decoder);
  KJNodes' tiny VAE for H3-Image (**C**).
- **TileShare** (2610.32233; abstract **B**): a training-free speedup of the tiled decoder.
- **vllm-omni #10106**, a 26-layer LightVAE decoder (**T**).

## 10. Refine and upscale (a possible stage after base or Pika renders)

- **Open-H3-Regenerate-2K** (sglang #47668, **C** body): a 768p draft to 2560x1440 by re-noise and three Euler steps,
  with audio held; it merges a Turbo LoRA. **Open-H3-Regenerate-2K-IC** (Hugging Face; card **B**, listing **H**): an
  in-context LoRA that keeps the draft as a reference video and outputs up to 4K.
- **Faces from less noise** (`BennyDaBall930/ComfyUI-Open-H3-FaceRegion`; **B**): a per-token strength map that starts
  face tokens from lower noise in the regenerate pass, through H3's per-token denoise mask, with a large identity
  claim. *For us:* our masked-source machinery already speaks per-token masks, so this is cheap to try in a remake or
  refine pass.
- **Other refine tools** (**B**): `JCKPIIIX/ComfyUI-MiniMaxH3-Refine` (low-denoise fl2va refine, with a denoise-to-sigma
  table at shift 12), `AnniCodex/ComfyUI-H3-FaceRefine-Plus` (an optical-flow anti-wobble lock), `bno84/ComfyUI-H3Regenerate`
  (a strict LoRA loader that fails on unmatched keys), 2D latent upscalers (`A1597404395` / `LBH-123-AI`, `SOLRICKS`),
  and `grawwww-studio`'s face-identity tests of refiners and distills.

## 11. LoRAs and training

- **DiffSynth training** (**C**):
  - full and LoRA recipes for H3 and H3-Image;
  - a LoRA script on the exact Comfy int8 H3-Image DiT and int8 encoder the owner runs
    (`examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Int8-Pruned-Comfy.sh`);
  - open #1824: H3-Image training currently encodes every caption as blank;
  - #1735: fine-tuning the CFG-distilled base comes out blurry, and DiffSynth trains video at a shift unlike 12/3;
  - #1798: sequence-parallel training (**B**);
  - the doc and the scripts disagree on the target modules.

  *For us:* the starting point if the owner trains an image or video LoRA, with the caption bug to avoid. Whether its
  saved keys load in `MiniMaxH3LoRABranch` is unchecked.
- **LoRA loading elsewhere.**
  - LightX2V modes `merge`/`separate`/`consumer`, with updates reaching AdaLN and the final modulation (**C**).
  - sglang stacked dynamic adapters (**C**), #50611 on AdaLN LoRA dtype on pruned files (**C**), and #43780, a shape
    check on 2-D `lora_B` (body **H**).
  - vllm-omni's native ComfyUI-layout loader (**C**), and #9238 / #9156 (**T**).
  - *For us:* #50611 raises a question for `lora_branch.py` on pruned files, unchecked.
- **SimpleTuner LoRA journal** (**B**, in progress).

## 12. ComfyUI core: what the pack copies, pins and patches

All are on [`../upstream/engines/comfyui_core.md`](../upstream/engines/comfyui_core.md).
- The `bench/check_reference_encode.py` pins hold at HEAD (**H**, run).
- `FinalLayer` / `_pdd_head` signatures and the `IMAGE_ENCODE_SEED` rename are already followed (**C**).
- A `lora_branch.py` docstring cites a function whose relayout moved (**C**).
- Three line citations in `ph3_image_nodes.py` have drifted (**H**). They are comment-only and left for a worktree
  edit.
- The repo slug is now `Comfy-Org/ComfyUI` (**H**).
- Open: #17715 (refiner every step), #17649 (encoder reference cycle), #17701 (malloc-graph arena VRAM), #16712
  (mask-driven packing), #17705 (time-embedder layout warning). All **C**.
- Release issue #85 (**C**; core's cast **H**): 17-frame brightness dips on unpruned files, traced to the
  time-embedder cast. Not on the shipped graphs, which load pruned files (reasoned).

## 13. Distills and ports (for awareness; the owner runs no fast distill except Pika)

- **Distills** (**B**): HyperFlow-H3-Flash (a compressed few-step H3 with audio, INT8 and GGUF), TurboH3, H3 AR
  few-step and 3-step LoRAs, CROWN HyperPDMD, and sglang's opt-in four-step SMD preset (#44739, **C** body).
- **Ports** (**B**): Apple Silicon (MLX, and an int8 Metal kernel claiming byte-identical output), AMD RDNA4, a Rust
  W4A4 engine, and a Hygon port. The Hygon port found the vendor's "flashattn" path was a bf16 matmul plus softmax
  that produced checkerboard noise.

## 14. Just fun

- `robertbschuette/MiniMax-H3-360-Orbit-LoRA`: fl2va with the same first and last frame makes a frozen-time orbit (**B**).
- arXiv 2611.08099: the H3 causal VAE adapted as a discrete tokenizer for 20 kbps video (abstract **B**).
- SolarRenderer (2610.28630) and AnyGameRenderer (2610.31466): game blockouts and structure videos turned into video
  through ref2va LoRAs (abstracts **B**).
- Duo 1.0 (vllm-omni #10366; body **H**): a full-duplex audio-visual world model whose rendering module is the H3 DiT.
- VA-TDM (2611.06796): adversarial distillation of H3 (abstract **B**).

## 15. SAM 3.1: what upstream changed, and what others do with it (`sam31_corrections.py`, `subject_track.py`, `docs/wiki/masked_v2v/sam3_prompting.md`)

**From Meta and core, nothing new since the pack's 2026-10-07 and 2026-10-18 reads.**
- **Meta's `coderef/sam3`, at its tip `eef6839` (2026-11-13; commits read H):**
  - `0570b3a` (2026-10-06) added an instance cap by area, a seed frame index and a backward box pass. It is already
    cited in [`masking/2026-10-07_mryolk.md`](masking/2026-10-07_mryolk.md). `subject_track.py` already tracks backward from the seed to
    the shot's start (`subject_track.py:94`) and picks one subject.
  - `74882c2` (2026-09-18) added a caller-mask request, `add_mask` with `force_tracker_propagation`. `subject_track.py`
    already notes it (`subject_track.py:16-23`): the 3.1 multiplex checkpoint does not serve it, and core's port lacks
    it, so the node does the same through core's `initial_mask`.
  - `f92a5ef` (2026-11-10) changes a type annotation in the training code only.
- **Core** (H): no commit touches `comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`, `comfy/ldm/sam3d_body` or
  `comfy_extras/nodes_sam3d_body.py` since 2026-10-01. So all three corrections in `sam31_corrections.py` (the -1..1
  image range, exact GELU, `memory_selection`) still apply as written.
- **The `phrase:N` syntax** (H). Core's tokenizer strips parentheses, splits on commas into separate detector passes,
  and keeps the top N per phrase, default 1 (`comfy/text_encoders/sam3_clip.py::_parse_prompts`, used by
  `comfy_extras/nodes_sam3.py::SAM3_Detect`). The pack already documents and handles this
  (`docs/wiki/masked_v2v/sam3_prompting.md`, `subject_track.py::counted`).

**Ways of using SAM 3.1 that the pack does not have** (MAINodes, community research; module headers and commit
messages read H, code not run; their figures are their claims):
*Corrected the same day after a Gemini review that read the code in full, checked by hand against the files cited
(`coderef/ComfyUI-MAINodes/snap_faces.py:1-12,156-162`; `coderef/ComfyUI-MAINodes/web/snap/edits.js:420-450`; `coderef/ComfyUI-MAINodes/snap_iris.py:62-71`; `bench/draw_face_points.py:1186-1192`):
items 1–3 below first got the face model, the edit's mask source and the eye tool wrong.*

1. **An edit kept only where SAM finds the named thing** (`1bf88d9`, 2026-11-07). SAM 3.1 runs with the words on
   Picture 1's crop, the image *before* the edit. The mask is grown and softened, then multiplied into the
   paste-back's feathered mask, so only that thing changes. It uses core nodes only.
   - Because the mask comes from the before image, it suits removals and in-place recolours of something already
     there (their case was taking a bar off a nose). An addition or shape change would be clipped by it and needs the
     after image's mask too.
   - *For us:* a word-driven mask source, used by two separate mechanisms. `MiniMaxH3EditGuider.mask` gates the
     edit's velocity difference per token in a two-pass edit, and is not a pixel lock. A pixel paste-back is the
     other. Accessory and held-item restores in the masked lane are the paste-back case.
2. **Small parts with guards** (`coderef/ComfyUI-MAINodes/snap_iris.py`, a custom node fed by core `SAM3_Detect` masks).
   - Irises come from two phrases combined, `"eye:2"` at threshold 0.6 and `"pupil:2, iris of the eye:2"` at 0.3,
     because on stylised faces either phrase alone misses.
   - Guards: a side-by-side pair (counted as two column runs across the whole frame, which a multi-person shot or a
     glasses rim can fool), an area cap, a minimum pixel count, and pass-through when there is no pair.
   - The iris pixels are shifted in Lab a/b toward the reference's median. A lightness weight and a guard near bright
     pixels keep the pupil, sclera and highlights from tinting, and a grey reference skips the step.
   - *For us:* the pack has no post-render eye-colour tool. `--iris contrast` draws irises on the grey body model
     before generation. If eye-colour drift shows on a swap, borrow the lightness weight and halo guard, but take the
     region from the subject mask and face points rather than full-frame runs.
3. **Identity notes** (`coderef/ComfyUI-MAINodes/snap_faces.py`).
   - **The face score is ArcFace** (IResNet-101, CVLface WebFace4M) on a square crop round each SAM 3.1 face mask.
   - DINOv3 ViT-H is an optional "character" score on a 1.5x unmasked box, added because ArcFace fails on anime
     profiles.
   - Duplicate face boxes are dropped, keeping the first. The separation figures are the author's, on their fixtures.
   - *For us:* ArcFace is the natural first identity metric for live-action casts (`docs/wiki/masked_v2v/who_is_who.md`,
     post-swap scoring). DINOv3 on an unmasked box mixes in hair, collar and lighting.
4. **A hair-lightness check**: the lightness of the lit part of SAM 3.1's hair mask, used to grade grey and silver
   edits (`coderef/ComfyUI-MAINodes/snap_faces.py`).
5. **Multi-person chains**: section 1 (`TheBeast-Spec/minimax-h3-multi-swap-comfyui`, **B**).

## 16. MAINodes and this pack: what came from where (a provenance ledger)

The two authors share ideas openly (Discord, each other's repos). This ledger dates each shared concept in both git
histories (`git log -S` / `-G`).
- **Timestamps.** Same-day order uses commit times: ours stamp `-0600`, theirs `-0500`, and both are US Central.
- **Reading a "theirs first".** Nothing in this pack mentions MAINodes before `2cca9748` (2026-11-04). For anything
  built here before that day, "theirs first" says nothing about whether we knew of their version.
- **Depth.** Read by a subagent (**C**). Spot-checked by hand (**H**): `ef91d35c` and `d5541e44` here, `ee3e435`
  there, the T5d table, and `docs/wiki/references.md:1767`.

| concept | here (commit, date) | MAINodes (commit, date) | direction and credit |
|---|---|---|---|
| post-refiner (`token_refiner` output) steering seam | `d5541e44` 2026-11-04 | `ace8169` 2026-10-30 (ported from their earlier ComfyUI-Dials) | theirs first, credited in our 2026-11-04 control-vector research |
| step windows for steering | `dcbec82d` 2026-11-04 | `f8a8c9a` 2026-10-31 (`coderef/ComfyUI-MAINodes/dials/envelopes.py`) | theirs first, credited ("a port of MAINodes'") |
| within-clip time curves | not built | `d11d438` 2026-11-01 (`coderef/ComfyUI-MAINodes/dials/timecurve.py`) | theirs, not adopted (planned in our 2026-11-06 comparison) |
| separate video and audio steering strength | `ef91d35c` 2026-11-06 14:06 | `d11d438` 2026-11-01 (`audio="protect"`) | theirs first, adopted; credited in our docs, not in code or commit messages |
| picture and video contrast vectors | `ef91d35c` 2026-11-06 | `2d160ae` / `cc00a34` 2026-10-31 | theirs first, adopted, credited in our docs |
| soft hold (ridge on `project_out`) | `fd4f246a` 2026-11-06 | rack hold `27563ac` 2026-10-31; DeTangle 2026-10-17 | theirs first, adopted, credited in our docs |
| size-matched placebo control | rule since 2026-08 (`docs/wiki/archive/2026-09-05_retracted_claims_part2.md`); `6bb1eaad` 2026-11-04 | `b8df909` 2026-11-01 (their LESSONS) | both; ours predates theirs. *Corrected the same day:* `6bb1eaad`'s commit message credits ComfyUI-Dials and MAINodes |
| `differing_span` pooling | `d5541e44` 2026-11-04; `877dc9ba` 2026-11-05 | `ee3e435` 2026-11-06 (idea); `dials_token_audit.py` | **ours first, credited**: they link our `control_vectors.py` and guide; their code is a separate reimplementation |
| leaving vision rows alone (`mask_vision`) | `4ecaa7df` 2026-11-04 | `ee3e435` 2026-11-06 | **ours first, credited** |
| excluding the sink token (row 0) | `bb72d811` 2026-11-05 | `ee3e435` 2026-11-06 | **ours first, credited** (they quote the owner's Discord report) |
| J-lens word vectors | `efbef735` / `87c0d1de` 2026-11-06 (12:59 / 13:33 -0600) | `ee3e435` 2026-11-06 13:39 -0500 (idea, credited); `833c02f` word dials | **ours first via Discord, credited**. Their idea commit precedes our first J-lens commit by about 20 minutes, so priority rests on the Discord report they cite, not on git order. Both follow `anthropics/jacobian-lens` on the encoder's 50-63 tail: shared method, not shared code |
| neutral word baseline | `87c0d1de` (`DEFAULT_NEUTRAL_WORDS`) | `833c02f` (centering on a bland list plus the mean of several lists) | unclear, possibly independent; the mechanics differ |
| per-kind condition strength around 0.999 | `c7fcae2f` 2026-10-08 (`reference_noise.py`, one level) | `4e957e2` 2026-09-20 (`coderef/ComfyUI-MAINodes/h3_cond_strength.py`, per guide, per reference, per stream) | theirs first, possibly independent (both read core and sglang); per-kind strength not adopted |
| releasing references mid-denoise | `ef91d35c` 2026-11-06; `441d46ba` 2026-11-11 (`MiniMaxH3ReleaseReferences`) | none (no step window in their H3 code) | ours, prompted by reading `h3_cond_strength` |
| token-grid mask max-pooling | `718dd3a5` 2026-10-04 (`video_mask.py`, space and time) | `bd785d3` 2026-08-13 (`coderef/ComfyUI-MAINodes/motion.py`, time only) | theirs first, possibly independent (both follow core's token clock) |
| `NestedTensor` video and audio noise masks | `74703635` 2026-08-13; `audio_freeze.py` 2026-09-12 | `2ecdff5` 2026-08-08 (`coderef/ComfyUI-MAINodes/motion.py`) | theirs first, possibly independent (both use core's API) |
| window loops | `audio_freeze_song.py` 2026-09-12; `chain_lengths`, `on_both_clocks` 2026-10 | `window_loop.py` 2026-08-12; `coderef/ComfyUI-MAINodes/h3_extend.py` 2026-08-23 | independent, different purposes (theirs re-renders to fix smear; ours chains on frozen audio along the dual clock) |
| SAM 3.1 | `sam31_corrections.py` 2026-10-07 | stock SAM 3.1 word masks from 2026-11-04 | independent; theirs carries none of our corrections |
| Pika preset | `b5539d95` 2026-11-04 | `6072b60` 2026-11-01 | independent (both from the Comfy-Org template) |
| steering on the image model | `67d1d705` 2026-11-08 (P-H3-Image); `c7c2b11b` 2026-11-12 (`J_ref_h3_image`) | `20c9114` 2026-11-12 | concurrent |

**Exist on one side only.**
- *Only here:* sound cues, sprite trajectories, the `old -> new` edit prompt, audio attention gain, the frozen-video
  K/V cache, subject tracking across cuts, Sapiens2, SAM 3D Body and clean-plate restore.
- *Only there:* colour carry, retiming (`coderef/ComfyUI-MAINodes/motion.py`, `H3RetimeAudio`), the block-state concept lab and the `diff_b`
  pre-refiner bias.
- *Both, designed differently:* reference sockets (their `h3_refs.py` bundles core's node; ours is the typed
  `AppendRef*` set), and refiner LoRA handling (our `safe_open` runner against their loaded MODEL).

**Their follow-up tests on our ideas** (T1, T3, T5, T6, word centering) are now read against the pack in section
17.2 (d), with the Gemini review's corrections.

**Housekeeping found, not changed.**
- `docs/wiki/references.md:1767`, written at 12:29 on 2026-11-06, still lists `audio="protect"` and the picture and
  video contrast dials as things this pack does not do. `ef91d35c` built both at 14:06 that day.
- `docs/wiki/references.md:140` names a third-party author by full name rather than by handle.
- MAINodes' credit lines show the owner's name in their visible link text. That is their repo's text, not ours.

## 17. Outcome first: what each side could take from the other

*Rewritten the same day at the owner's request. The first version of this section and the end of section 16 were
feature lists: they matched nodes to problems. This version starts from what each side is trying to achieve.*

**Sources.**
- Two read-only subagents read MAINodes beyond Snap, Motion Lab, low VRAM, RelayVSR and the dials research. Their
  load-bearing citations were checked by hand: `coderef/ComfyUI-MAINodes/dials/research/PLAN.md:26,34`, `coderef/ComfyUI-MAINodes/ALPHA.md:69,149`,
  `coderef/ComfyUI-MAINodes/capsule_types.py:86`, `coderef/ComfyUI-MAINodes/TUNING.md:162`, `coderef/ComfyUI-MAINodes/h3_extend.py:112,183`, `coderef/ComfyUI-MAINodes/docs/snap/AGENT_START.md:202`,
  `coderef/ComfyUI-MAINodes/web/snap/checks.js:91`, and `coderef/ComfyUI-MAINodes/docs/snap/CLOUD_HANDOFF_2026-11-08.md:30`.
- A Gemini review the same day read MAINodes' SAM, steering and T-series code in full, and ran a cosine check on the
  pack's own vectors. Its corrections are folded in and marked. Its measurements are its own and were not re-run here.

**Labels.** Their figures are their claims. **H** means checked by hand; otherwise the line is a subagent's or Gemini's
reading.

**Whose outcomes.**
- **The pack's** (`VISION.md`, `docs/wiki/projects/index.md`, the owner's working rules): the best quality and speed
  one RTX 4090 can reach, as claims a later reader can re-derive. Per track:
  - (a) masked pieces with every visible subject replaced and identity holding on the owner's playback;
  - (b) long song videos over frozen audio;
  - (c) image edits from the phone studio;
  - (d) steering the owner can trust;
  - (e) speed at quality on the card;
  - (f) less of the owner's time spent reviewing.
- **MAINodes'**, in their words:
  - Motion Lab: fast motion without smear, beats that don't multiply when time is dilated, invisible seams.
  - Dials: "more of the intended change per unit of collateral change".
  - Snap: one person directing agents through a short film, where "consistency is data, not prompting", with the
    person's eye as the judge.
  - Also: running on smaller cards, and upscaling.

### 17.1 The largest thing to borrow is method, not a node

These practices serve every outcome above, ours and theirs.
- **A gate written before the run.** Their `coderef/ComfyUI-MAINodes/dials/research/PLAN.md` queue has a column "gate (what decides the next
  step)" (**H**, `:34`), filled in before an experiment runs, with one page per experiment saying what it changed in
  the plan. The pack has dated records and controls, but no pre-registered decision column.
- **An evidence ladder on every number.** "One seed is a hint, two seeds a finding, held-out scenes a result"
  (**H**, `:26`). It is a compact form of the pack's evidence labels that a reader can apply at a glance.
- **An identity setting on every node that changes the model.** Their pass-through settings reproduce stock exactly
  (`torch.equal`): `jerk passthrough`, DyRoPE's `compact_all`, uniform condition strength. An A/B is then one widget,
  not two graphs.
- **Refuse rather than guess.** Their plan estimate returns `-1` "because an uncalibrated guess is worse than no
  answer". The pack's Preflight prices exactly. The same rule fits any estimate we add later.
- **Splices that check themselves.** Their repair splice asserts zero difference outside the spliced range.
- **Every reviewer miss becomes a test.** Snap's critic loop:
  - every miss the person catches is "one press from a fixture" (SNAP-R108);
  - a critic prompt ships only if it holds a floor of caught flaws and false alarms on clean pictures (SNAP-R109);
  - the floor is written down, and their latest blind run fell below it and stays open (**H**, `coderef/ComfyUI-MAINodes/docs/snap/CLOUD_HANDOFF_2026-11-08.md:30`).

  This is the single most direct lever on outcome (f). It turns the owner's catches on playback into a measured
  reader, instead of a reader the owner has to re-check.
- **Stop rerolling, change the approach.** "Two or three failed seeds of one recipe stop that recipe"; only a taste
  choice between results that work goes to the person (SNAP-R200). A rejection is kept with why and when to look again
  (their "cemetery"). This matches the owner's ask-once rule and could stop rerolls before they reach the owner.
- **Keep the negatives.** "Explicitly not doing, with reasons", published corrections, and an examples archive that
  still runs.

**Where the pack is ahead on method, and what it would give them.**
- Claim labels that point at the command or constant behind a figure (`VISION.md`, `docs/wiki/ops/evidence.md`).
  Many of their figures cite no command, are small-n on one machine, or point to LESSONS files that are not in the
  public repo. Their own reviewer agent flagged that one critic floor mixed subset reruns with earlier results.
- Vendor-parity numerics, and the upstream sweep machinery (`docs/wiki/upstream/`). They track the same core PRs by
  hand.
- Review on playback. Their critic reads stills and tiles, and their closeout says so ("sampled visual … not continuous
  perceptual playback"). The pack's stacked review clips, read by a model that watches and listens natively, close
  that gap.

### 17.2 The pack's outcomes, and what in their world moves each

- **(a) Masked pieces with identity on playback.**
  - *Identity as data before the owner looks.* ArcFace notes per face (section 15, corrected) could list the frames
    worth looking at per subject. That is a list, never a verdict, under the owner's rule; their thresholds were set on
    anime fixtures and need re-measuring on live action.
  - *Remake a few frames, not a whole load.* Their Repair Plan/Splice snaps a bad range outward to whole tokens, ends
    at a shot cut, asserts the rest unchanged and reports the seam against the clip's own frame delta.
  - *Their mask lesson:* "The mechanisms were fine; the mask author wasn't."
  - *A trap for us:* on masked runs, per-step `x0` "gives you a picture of the mask", which bears on
    `MiniMaxH3StepX0Observer` readings.
- **(b) Long song videos.**
  - *An answer to an open question here.* The pack lists "two ways to continue a clip … seams uncompared"
    (`docs/wiki/core/h3_audio_freeze.md:445`, **H**). Their extension work found the masked latent prefix beat the
    image guide "everywhere we measured" (**H**, `coderef/ComfyUI-MAINodes/ALPHA.md:69`), on two content sets.
  - *Texture drift.* "Standard grows a mosaic texture past ~40 s; the re-anchor alone holds 2 minutes" (SNAP-R64; the
    same author's `h3-long-video-research`). This is a check to run on the pack's longest chains.
  - *Choosing the context window.* A 90-frame handle overlapped better than 39, which bears on the pack's context-39
    or context-90 choice.
  - Their colour carry does not answer VAE round-trip darkening here, since the pack's context never makes a VAE round
    trip (`audio_freeze.py`, the latent prefix copy; reasoned). *Nuance from the second Gemini review:* if long chains
    drift in exposure or saturation from the DiT alone, `coderef/ComfyUI-MAINodes/h3_color_carry.py`'s
    `z + E(corrected) - E(original)` is the way to pull a latent prefix back toward a pixel-space colour target while
    cancelling the encoder's own bias.
- **(c) Phone-studio image edits.**
  - Snap's model-choice table and its new-view traps give edit-route defaults: region crop with paste-back, crop past
    thin crossing objects, a new picture for a new look, and showing now and after side by side.
  - Their conditioning banks persist to disk across workflows and restarts, where the pack's split cache lives in
    ComfyUI's memory.

  For the studio session to consider, through its own session (memory: app changes go through it).
- **(d) Steering the owner can trust.** *Partly corrected after the Gemini review; the pack already had more of this
  than the first version said.*
  - *T3, span pooling at matched size:* the pack already has this control, with norm-matched arms at one target norm
    across pooling modes (`bench/results/2026-11-05_control_vectors_pika_and_ref2va.md:117-119`, **H**). Their "no
    clear gain" came on short one-line pairs. On the pack's long structured prompts, whole-scene pooling dilutes the
    vector, and `MiniMaxH3Steer` already norm-matches by default.
  - *T6, the family check:* Gemini's cosine run on the pack's own Pika fl2va and ref2va vectors found them nearly
    coincident (its measurement, not re-run). Also, `control_vector_nodes.py::resolve_vector_file_path` already swaps
    in the sibling family's file, and refuses only when none exists (**H**). The MiniMax-H3-Image refiner differs
    markedly from the video ones in the same run. That supports keeping `h3_image` its own family, and may be why
    MAINodes' image dials needed several times the video strength (Gemini's inference).
  - *T5, a first frame weakens steering:* for plain fl2va image-to-video, the pack can already lower the first frame's
    hold through `MiniMaxH3ReferenceNoise` or `MiniMaxH3ReleaseReferences` `clean_level`.
    - Splitting guide strength from reference strength runs into core labelling both segments from one scalar
      (`comfy/ldm/minimax/model.py`, `seg_t["cond"]` and `seg_t["ref_img"]` both from `vis_aug`, **H**).
    - Their notes add that the first frame also enters the encoder as clean vision tokens.
    - A step-scheduled release of the first frame, which they did not test, is the candidate Gemini proposes.
  - *Word dials:* the pack's `j_lens.py` already centres across its whole word vocabulary and subtracts a neutral
    list. Their twelve-list mean depends on the batch, as their own notes say. What to take from them is the placebo:
    a bland list at +2 still changed the clip, so word-dial checks should include a loudness check.
  - *Lessons that apply directly:*
    - sum dials once in float64;
    - a blend level outside 0–1 extrapolates rather than strengthens, which bears on `MiniMaxH3EditGuider` scale above 1;
    - audio kept on the undialled branch still moves, because audio attends to the dialled video, which bears on
      `audio_scale`;
    - "a strong direction is not isolated control".
- **(e) Speed at quality on the card.**
  - Their streamed blocks are bit-exact by mechanism (int32 accumulate), with a first-block self-check, and four
    ComfyUI flags cut host RAM sharply. Both fit the pack's memory-bound long renders. They were measured on a
    machine fenced down to a smaller card.
  - Their gate keeps a fixtures manifest with a sha256 per file and a stale-result guard, so a crashed suite prints
    FAIL, never the last PASS.
  - Nothing for Sol.
  - *Delivery:* RelayVSR (`coderef/ComfyUI-MAINodes/relayvsr/README.md`) reports that its X4 model downscaled to 2x
    beats its own X2 on H3 clips, that a back-projection preset cuts flicker without redrawing faces, and that it fits
    a card of the 4090's size. All their figures, on five short clips. It is a candidate final upscale for 1344x768
    renders.
- **(f) Less owner review time.**
  - The critic loop and the reroll stop rule (17.1).
  - *Triage notes, never verdicts.* Snap keeps approval with the person and adds cheap measurements as notes, so the
    person looks only where a note points (`coderef/ComfyUI-MAINodes/snap_critiques.py`,
    `coderef/ComfyUI-MAINodes/snap_cards/critic.json`):
    - Voice QA: speaker timbre, median F0 and transcribed words with timestamps (`coderef/ComfyUI-MAINodes/snap_voice.py`).
      It catches voice drift and unscripted speech, and partners with `bench/voice_spans.py` and `audio_end_percent`.
    - A copy check: a DINOv3 whole-frame score that flags a take tracing `<Picture 1>` instead of moving.
    - Face and hair notes per face (section 15).

    This matches the owner's rule that a reading lists what to look at and holds nothing.
  - Their Video Compare loops a bracket over a seam, flickers A against B, and carries a starred winner into the next
    queue.
  - One caveat: Snap's one-question-at-a-time Room runs against the owner's rule to batch reviews.

### 17.3 Their outcomes, and what in the pack moves each

- **Seams you cannot hear.** Their ambience "still steps at the cut", only a 65-tick audio prefix is carried, and they
  write that "only a pinned master track GUARANTEES the structure". *Refined after the second Gemini review:* two cases.
  - **Fixed-speed extension** (`coderef/ComfyUI-MAINodes/h3_extend.py`). The pack's whole-slice frozen audio
    (`MiniMaxH3FreezeAudio`, `MiniMaxH3AudioFreezeSong`) is the pinned track. Its run-phase check
    (`(latent_t - context_steps) % 5 == 0`, `audio_freeze.py:378`, **H**) keeps the copied video prefix on a
    clip-start step.
  - **Motion Lab's dilated passes.** Freezing the *baseline* audio would break sync, because the second pass runs on a
    dilated clock. What transfers is freezing their already smeared track (their H3 Audio Smear, which today only
    seeds the audio) with the same `NestedTensor` mask across the rolling windows. Reasoned, not tested.
  - *Corrected the same day:* they do know the dual-clock rule (`coderef/ComfyUI-MAINodes/h3_extend.py:112,183`; `coderef/ComfyUI-MAINodes/capsule_types.py:86`; **H**).
    Their extension tool notes an off-grid length, and `coderef/ComfyUI-MAINodes/motion.py` pads to any 17k+5. The pack refuses off-grid chains
    (`h3_rules.on_both_clocks`).
- **Beats that don't multiply.** Their planned fix is sparse pose anchors.
  - Core's `MiniMaxH3AddGuide` already gives them mid-clip frames (their ROADMAP §4). What the pack adds is releasing
    them partway through denoising, through `MiniMaxH3ReleaseReferences` `guide_end_percent`
    (`control_vector_nodes.py:4763`, **H**). *Corrected the same day: this line put `guide_end_percent` on
    `MiniMaxH3GuideLatent`.*
  - `MiniMaxH3SoundCues` anchors impacts on the exact audio grid.
  - `MiniMaxH3BodyMeshVideo` could supply the pose ladder.
  - Their oracle's inability to abstain has nothing on our side.
- **Dials that keep working.**
  - **Image dials and the image refiner.** Their `MiniMaxH3ImageModel` dials reuse the video text path, and they
    needed several times the video strength. Gemini's run on the pack's refiner Jacobians found the image refiner
    rotated well away from the video ones. That plausibly explains the strength gap; the vector norms differ too.
    The pack's `safe_open` refiner runner and J-lens re-projection (`j_lens.py`) carry one encoder-space vector into
    the image family without loading the DiT.
  - **Core PR #17715 is a compute heads-up, not a breakage.** *Corrected the same day, after the second Gemini
    review, checked by hand.* Their `ContextShift` calls `preprocess_text_embeds` itself
    (`coderef/ComfyUI-MAINodes/dials/core.py:110`). Core skips the refiner when the input is already the model's width
    (`comfy/ldm/minimax/model.py`, the `text_states.shape[-1] != self.hidden_size` branch). Their `PostShift` hook on
    `token_refiner` is the fallback. Under #17715 the refiner would run on every step. It is on
    `../upstream/horizon.md`.
- **Snap's consistency.**
  - Their checks run stock SAM 3.1 (`SAM3_Detect`, and `SAM3_VideoTrack` with one object; **H**, `coderef/ComfyUI-MAINodes/web/snap/checks.js:91`).
    `MiniMaxH3SAM31Corrections` targets that path, though their thresholds would need re-measuring after it.
  - Their Transform Footage reports a capability gap for frame ranges and temporal masks. The pack's
    `MiniMaxH3MaskedSource` and `MiniMaxH3RestorePlate` are that capability.
  - Their H3 setups don't wire a last frame (F-F20, **H**, `coderef/ComfyUI-MAINodes/docs/snap/AGENT_START.md:202`). `MiniMaxH3Conditioning` handles a
    lone last frame.
- **Low VRAM and big canvases.**
  - Their token counts are video-only approximations, and nothing in their files names the `int32` QKV line. Their
    longest runs, around 217k tokens at 1376x768, sit past that line and past the pack's trained-family area cap
    (**H**, `coderef/ComfyUI-MAINodes/ALPHA.md:149`). `MiniMaxH3Preflight` prices exactly.
  - Their motion and turbo LoRAs load merged on int8. Some of their motion loss may be merge requantisation, which
    `MiniMaxH3LoRABranch` avoids (cannot tell from their files).
- **Unscripted speech from voice references.** `MiniMaxH3ReleaseReferences` `audio_end_percent` might reduce how much
  H3 copies a reference's words. Untested.

### 17.4 If any of this is passed along

**To them, in order** (through the owner; nothing has been sent; reordered after the second Gemini review):
1. The image refiner's divergence from the video ones, which bears on their image-dial strength. With it, the pack's
   refiner runner and J-lens re-projection.
2. The SAM 3.1 corrections, after reading what the other session's SAM 3.1 worktree changes.
3. Frozen audio with the run-phase check, for their extension seams. For their dilated passes, freezing the smeared
   track instead.
4. `MiniMaxH3LoRABranch` for their merged int8 LoRAs, and the `int32` QKV line for their long dilated runs. The pack's
   `MiniMaxH3SageAttention` fork has `int64`-safe indexing, relevant because their stack lists sage. The owner no
   longer uses sage.
5. Smaller items:
   - the #17715 compute heads-up;
   - `MiniMaxH3ReleaseReferences` (`guide_end_percent`, `audio_end_percent`);
   - `MiniMaxH3EncodeReferences` for Snap's prompt revisions;
   - `MiniMaxH3MaskedSource` and `MiniMaxH3Conditioning`'s lone last frame, for Transform Footage.

**For us, in order:**
1. A gate column and the seed ladder in the pack's research plans.
2. A critic or reader floor fed by the owner's catches, with triage notes (voice, copy, face) feeding it.
3. Their masked-prefix seam result, read against the song-loop handoff.
4. ArcFace identity notes as a frames-to-look-at list.
5. RelayVSR as a candidate final upscale.

### 17.5 Not read

- In Snap: most of `SNAP_RULES.md`, the director code, the suite code and most dogfood reports. The `docs/snap/`
  DECISIONS, LESSONS and PLAN files their `AGENTS.md` points to are not in the clone.
- Outside Snap: `flow/`, `DEROPE_ANY_MODEL.md`, `docs/WINDOW_MODE.md`, the RelayVSR and concept-lab docs beyond their
  READMEs (the second Gemini review read the RelayVSR README, `snap_voice.py`, `snap_critiques.py` and
  `h3_color_carry.py`), and most of `coderef/ComfyUI-MAINodes/motion.py`, `h3_drift.py`, `h3_measure.py` and `video_compare.py` beyond headers.

---

## What was not dug into

- **Code of most periphery items.** Every node pack, Hugging Face card and paper above is **B** or **T**. None of
  their code was run or read in depth.
- **Titles only:** vllm-omni #9420, #7558, #7507, #9238, #9156, #10106, #10257 / #10179 (HyperFlow), #8677, #9721;
  sglang #50247, #49599, #50120, #50259, #43220, #44806, #44807; LightX2V's open PRs (#1391, #1528, #1550, #1558, #1574,
  #1554); DiffSynth #1790, #1823, #1563; core #17696, #16342, #16150, #17683, #17624, #17572.
- **Not checked against our code:**
  - where core builds the vision RoPE table;
  - whether core's encoder attention reaches cuDNN SDPA;
  - whether core builds the audio VAE's buffers under a bf16 default;
  - how `lora_branch.py` treats AdaLN keys on pruned files;
  - whether DiffSynth's LoRA keys load in `MiniMaxH3LoRABranch`;
  - our Pika condition-noise draw order against the three engines;
  - `H3_IMAGE_MANUAL_SIGMAS` against the ulp-shifted sigmas;
  - which H3-Image reference noise order the release uses;
  - the Hub-hosted H3-Image-PE blocks;
  - whether ArcFace face embeddings separate the masked lane's live-action cast (MAINodes' SAM and identity code
    was read in full by the Gemini review, section 15);
  - MAINodes' T-series beyond what section 17.2 (d) records.
- **Searched, nothing found:** H3 issues in kijai's WanVideoWrapper since 11-01; arXiv for Qwen3-VL as a diffusion text
  encoder since 10-15. Hugging Face search caps at 100 results per term, so small uploads may be missing.
- **Nothing was rendered.** The only check run was `bench/check_reference_encode.py`.

## If one session picked three things to look at next

1. **Long clips.** The latent-carry finding (section 2) read against the song-loop handoff. It needs no render to
   read, and one pair to test.
2. **Remakes.** Faces from less noise (section 10) in a masked remake, which our per-token masks can already express.
3. **The image lane.** The velocity-form Euler step and soft-start (section 4), one cheap render each.

Added from sections 15 and 16, for whoever picks up steering or the masked lane next:
- **Steering.** A step-scheduled first-frame release for image-to-video steering (section 17.2 (d)). The T6 check
  is done (Gemini's run).
- **The masked lane and image edits.** The SAM word-gated paste (core nodes only, for removals and recolours), and
  ArcFace identity notes as a frames-to-look-at list. Section 17.4 has the fuller order.
