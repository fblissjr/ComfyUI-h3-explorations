# Upstream horizon: open work we are waiting on

last updated: 2026-11-17 (DiffSynth #1824 moved to "Landed or closed", after a peer session's review); 2026-11-15 (page created from [`sweeps/2026-11-15.md`](sweeps/2026-11-15.md); a second pass over sglang's and vllm-omni's open H3 PRs added the same day)

Open PRs, issues and announced work upstream, each with what it would change for this pack. A row stays until it
lands or closes. Then it moves to "Landed or closed" with the commit and the sweep that saw it. The status column is
as of the sweep in the row's "seen" column; re-check before acting. Ranking and trust:
[`index.md`](index.md), "When upstreams disagree".

## Open

| upstream | item | what it does | what it would touch here | seen |
|---|---|---|---|---|
| core | [#17715](https://github.com/Comfy-Org/ComfyUI/pull/17715) / [#17713](https://github.com/Comfy-Org/ComfyUI/issues/17713) | runs the H3 token refiner inside every step and casts its q/k to `v.dtype` | `lora_branch.py` refiner tooltip, `nodes.py` `patch_token_refiner`, the post-refiner seam the control vectors read | 2026-11-15 |
| core | [#17705](https://github.com/Comfy-Org/ComfyUI/pull/17705) / [#17692](https://github.com/Comfy-Org/ComfyUI/issues/17692) | warns on H3-Image checkpoints in the full time-embedder layout (reported as pure noise) | `bench/convert_h3_image_dit.py` writes that layout; ours rendered correctly (`bench/results/2026-11-11_h3_image_weights.md`) | 2026-11-15 |
| core | [#17649](https://github.com/Comfy-Org/ComfyUI/pull/17649) | breaks a reference cycle that keeps a dropped text encoder's weights allocated | host and VRAM headroom when the 32B encoder unloads | 2026-11-15 |
| core | [#17701](https://github.com/Comfy-Org/ComfyUI/pull/17701) | malloc-graph arena VRAM between steps | `frozen_video_cache.py` (checks `malloc_graph_enabled`) | 2026-11-15 |
| core | [#16712](https://github.com/Comfy-Org/ComfyUI/pull/16712) | packing driven by masks instead of contiguous reference-then-target rows | `control_vector_nodes.py` released layouts (interaction not traced) | 2026-11-15 |
| sglang | [#47668](https://github.com/sgl-project/sglang/pull/47668) | Open-H3-Regenerate-2K: 768p draft to 2560x1440 by re-noise and three Euler steps, audio held | a possible upscale stage after base or Pika renders (it merges a Turbo LoRA in) | 2026-11-15 |
| sglang | [#50611](https://github.com/sgl-project/sglang/pull/50611) | keeps an AdaLN LoRA's dtype when folding it into the pruned AdaLN space | how `lora_branch.py` treats AdaLN keys on pruned files | 2026-11-15 |
| sglang | [#43313](https://github.com/sgl-project/sglang/issues/43313) | Q4 roadmap: RTX 4090 default commands, FP8 attention and W8A8 on SM89, ComfyUI-speed parity | a vendor-path baseline on the owner's card | 2026-11-15 |
| sglang | [#45180](https://github.com/sgl-project/sglang/pull/45180), [#49033](https://github.com/sgl-project/sglang/pull/49033), [#48890](https://github.com/sgl-project/sglang/pull/48890) | online FP8 video VAE decoder; reference VAEs loaded only when used; a LoRA merge mode "disabled" | VAE decode precision; memory | 2026-11-15 |
| vllm-omni | [#9777](https://github.com/vllm-project/vllm-omni/pull/9777) (draft) | Pika H3 Flex with separate RNG streams and an option to keep the official bf16 rounding points; claims near-identity with the release on t2va/fl2va | seed-exact Pika parity (`h3_config.PIKA_*`) | 2026-11-15 |
| vllm-omni | [#9762](https://github.com/vllm-project/vllm-omni/pull/9762) | a ref2va video reference conditions only the first k steps, then the layout drops it | `MiniMaxH3ReleaseReferences` `video_end_percent` | 2026-11-15 |
| vllm-omni | [#9689](https://github.com/vllm-project/vllm-omni/pull/9689) / #9687 | opt-in reference isolation: reference queries read only reference K/V, captured once (not lossless, by its author) | `MiniMaxH3FrozenVideoCache` | 2026-11-15 |
| vllm-omni | [#10400](https://github.com/vllm-project/vllm-omni/pull/10400), [#10405](https://github.com/vllm-project/vllm-omni/pull/10405) | Heun and DPM++ 2M for H3, ported from ComfyUI's k-diffusion | sampler choice for base H3 | 2026-11-15 |
| vllm-omni | [#10256](https://github.com/vllm-project/vllm-omni/pull/10256) | batching VAE tiles changes fp16 bits | tiled decode parity | 2026-11-15 |
| vllm-omni | [#9398](https://github.com/vllm-project/vllm-omni/pull/9398) (draft) | unlocks non-24 fps output into the model clock | against the pack's fixed 24 fps rule | 2026-11-15 |
| vllm-omni | [#9863](https://github.com/vllm-project/vllm-omni/issues/9863), [#10307](https://github.com/vllm-project/vllm-omni/issues/10307) | asks for P-H3-Image; asks for Pika LoRA loading | a second engine for those two lanes | 2026-11-15 |
| vllm-omni | [#7380](https://github.com/vllm-project/vllm-omni/issues/7380) | RFC: H3 API and ComfyUI parity checklist | | 2026-11-15 |
| diffusers | [#15299](https://github.com/huggingface/diffusers/pull/15299) | loads the pruned ComfyUI checkpoints (curve interpolated as core does) and single-file H3-Image | a diffusers reference run on the owner's own files | 2026-11-15 |
| diffusers | [#15209](https://github.com/huggingface/diffusers/pull/15209) | Comfy int8 ConvRot checkpoints through comfy-kitchen | same | 2026-11-15 |
| diffusers | [#15199](https://github.com/huggingface/diffusers/pull/15199) / [#15245](https://github.com/huggingface/diffusers/issues/15245) | ref2va soundtracks not at 32 kHz without torchaudio | `MiniMaxH3AppendRefAudio` resampling (compare) | 2026-11-15 |
| DiffSynth | [#1735](https://github.com/modelscope/DiffSynth-Studio/issues/1735) | SFT on the CFG-distilled H3 base comes out blurry; training shift differs from 12/3 | H3 video LoRA training | 2026-11-15 |
| Pika | [#9](https://github.com/Pika-Labs/Pika-H3-Flex/issues/9) | how the distill was trained (LoRA or full, training shift, what the adapter holds) | the Pika contract's unknowns | 2026-11-15 |
| release | [MiniMax-H3 #91](https://github.com/MiniMax-AI/MiniMax-H3/pull/91) (draft) | Level-of-Token mixed-resolution DiT; research checkpoints below dense quality by its author | attention cost | 2026-11-15 |
| release | [MiniMax-H3 #85](https://github.com/MiniMax-AI/MiniMax-H3/issues/85) | 17-frame brightness dips on unpruned checkpoints from the time-embedder cast | `H3_IMAGE_UNET_BF16` and any unpruned video file (not the shipped graphs) | 2026-11-15 |
| vllm-omni | [#5899](https://github.com/vllm-project/vllm-omni/pull/5899) (draft) | runs the token refiner once per request instead of every step (no timestep dependency), claimed bit-identical; the opposite direction from core #17715 | the pack's once-before-step-0 refiner assumption (`lora_branch.py`, `nodes.py` `patch_token_refiner`) has an engine on its side | 2026-11-15 |
| vllm-omni | [#6779](https://github.com/vllm-project/vllm-omni/pull/6779) | sliding windows past 15 s: each continuation is a first-frame request from the decoded handoff frame, with the previous window's tail audio as a frozen audio reference | chained song loops; a decode-and-re-encode handoff, which community notes link to colour creep (`periphery.md`) | 2026-11-15 |
| vllm-omni | [#5952](https://github.com/vllm-project/vllm-omni/pull/5952) (draft) | calibrated TeaCache for ref2va at 50 steps (full compute, then a few cache hits) | base ref2va speed | 2026-11-15 |
| vllm-omni | [#9222](https://github.com/vllm-project/vllm-omni/pull/9222) | AlignCache: predict selected forwards from cached history, for H3 t2va | base speed | 2026-11-15 |
| vllm-omni | [#9728](https://github.com/vllm-project/vllm-omni/pull/9728) (draft) | reference-K/V refresh scheduled around cache hits, with a maximum staleness | `MiniMaxH3FrozenVideoCache`-style reuse | 2026-11-15 |
| vllm-omni | [#9959](https://github.com/vllm-project/vllm-omni/pull/9959) (draft, tracking only) | a ref2va latency ceiling (device-resident reference K/V, graded sparsity in the dense head/tail window); its author says it fails the quality bar | what the speed-quality frontier looks like when pushed | 2026-11-15 |
| vllm-omni | [#8813](https://github.com/vllm-project/vllm-omni/pull/8813) (draft) | audio VAE's transposed convolutions rewritten as phase-decomposed conv1d (Ascend motive) | audio decode, if it is ever slow here | 2026-11-15 |
| vllm-omni | [#10366](https://github.com/vllm-project/vllm-omni/pull/10366) (draft) | Duo 1.0, a full-duplex audio-visual world model whose Realization module is the H3 DiT | periphery: H3 as a world-model backbone | 2026-11-15 |
| vllm-omni | [#9420](https://github.com/vllm-project/vllm-omni/pull/9420), [#7507](https://github.com/vllm-project/vllm-omni/pull/7507) (draft) | multiframe reference workflow; Fun ControlNet Union workflow (titles only) | masked lane, control | 2026-11-15 |
| sglang | [#49080](https://github.com/sgl-project/sglang/pull/49080) | NVENC encode of the output MP4 (opt-in), audio muxed after | save time after decode | 2026-11-15 |
| sglang | [#37998](https://github.com/sgl-project/sglang/pull/37998) | optional `taeh3` tiny decoder for low-latency previews | live preview while sampling | 2026-11-15 |
| sglang | [#35803](https://github.com/sgl-project/sglang/pull/35803) | makes the ref2va still short edge configurable; its table shows the token cost of 2048 against smaller edges | `topics/reference_sizing.md` | 2026-11-15 |
| sglang | [#33552](https://github.com/sgl-project/sglang/pull/33552) | sigma schedule off-by-one; probably overtaken by the merged `32f34e898b` (#38671, `linspace(1, 0, N + 1)`), not checked | | 2026-11-15 |

## Landed or closed

| upstream | item | how it ended | seen |
|---|---|---|---|
| sglang | [#50218](https://github.com/sgl-project/sglang/issues/50218) | the SGLD app's SHA pin replaced by a symbol check (`6de483b326`, #50223) | 2026-11-15 |
| DiffSynth | [#1824](https://github.com/modelscope/DiffSynth-Studio/pull/1824) | the PR was closed unmerged, and the same fix (H3-Image training encoded every caption as `" "`) landed on main as `c05b58d6` (2026-11-16), crediting the PR's author. So an H3-Image LoRA trained with DiffSynth after that date used its captions | 2026-11-17 |
