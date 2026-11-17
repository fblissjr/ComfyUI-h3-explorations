# The periphery: how it differs, aligns, and what it does that others do not

last updated: 2026-11-17 (DiffSynth and LightX2V H3-Image inference paths read stage by stage, [`topics/h3_image_pipeline.md`](topics/h3_image_pipeline.md); the t2i vision-patch idea set against the engines); 2026-11-16 (added `geiru_context_suite`, `NerdNotes`, `ComfyUI-MMH3Tools`, and `ComfyUI-MiniMaxH3-Context-Loop` from hand audit); 2026-11-15 (page created; state as of [`sweeps/2026-11-15.md`](sweeps/2026-11-15.md))

Everything outside sglang, vllm-omni, diffusers and core: the other libraries, distill releases, node packs,
Hugging Face uploads and papers. **Periphery is never a vote** ([`index.md`](index.md), "When upstreams disagree").
It is read for where it differs from the ranked engines, where it aligns with them, and what it does that none of
them do. **Community custom nodes are community research:** interesting, trusted for nothing until something of ours
checks them, and their numbers are their claims.

This page holds the standing picture per source. Each sweep record has the dated detail.

## Libraries and distill releases

| source | aligns | differs | novel | where |
|---|---|---|---|---|
| DiffSynth-Studio | H3-Image bounds, labels and reference level match sglang; encoder at layer 50 pre-norm; loads Comfy-Org's pruned int8 H3-Image file. **H3-Image inference (2026-11-17):** text to image sends the prompt's tokens only; an in-bounds reference is cropped and its posterior drawn from the request's generator and rounded to fp16; a request below `MINIMAX_H3_IMAGE_MIN_GENERATION_AREA` is generated larger and resized back with Lanczos; velocity Euler on the six-step grid | enlarges ref2va stills to 2048; H3-Image training at about 1 MP with every row, references included, at the sampled timestep; trains video at shift 2.22, not 12/3 (#1735). **H3-Image inference (2026-11-17):** the shipped examples load every model at `torch.bfloat16`, the release's fp32 parts included (the VAE's latent statistics too, the agent's inference from `Module.to`); draw order is target, then per reference posterior then anchor; an empty prompt is encoded as `" "`; the output size is the request's (default 2048x2048), never the picture's | **training**: full and LoRA recipes for H3 and H3-Image, including a LoRA script on the Comfy int8 H3-Image DiT (`examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Int8-Pruned-Comfy.sh`). Open bug #1824: captions encoded as `" "` | `coderef/DiffSynth-Studio/diffsynth/`, `examples/minimax_h3*/` |
| LightX2V | Pika six-step contract, same knots, shifts and seeds as `h3_config.PIKA_*`; H3-Image 4 MP floor and Lanczos down. **H3-Image inference (2026-11-17, `model_cls="minimax_h3_image"`, tasks `t2i` and `i2i`):** text to image sends the prompt's tokens only; Qwen and VAE copies bounded as sglang bounds them (`coderef/LightX2V/lightx2v/models/input_encoders/hf/minimax_h3_image/preprocess.py::_bound_short_edge`); an in-bounds reference is cropped, its posterior drawn before the even crop and rounded to fp16; fp32 VAE encoder, bf16 decoder with fp32 norms (`_keep_in_fp32_modules`); velocity Euler | enlarges ref2va stills (`diffusers` mode). **H3-Image inference (2026-11-17):** draw order is every posterior, then target, then every anchor (its comment: Modular Diffusers' order); its shipped config runs the DiT on `sage_attn2` (quantized attention) with an AdaLN cache on; the image path refuses LoRA and a quantized DiT (agent's reading) | LoRA modes `merge`/`separate`/`consumer`, with updates reaching AdaLN and the final modulation; causal and streaming variants; a ComfyUI front end driving a LightX2V server (for H3-Image an HTTP client only, adding no numerics). An offline AdaLN modulation cache keyed on the timestep's fp32 bits (`networks/minimax_h3/adaln_cache.py`) | `coderef/LightX2V/configs/minimax_h3*`, `lightx2v/models/` |
| Pika-H3-Flex | the contract the pack runs ([`topics/pika_parity.md`](topics/pika_parity.md)) | | keeps 51 of 64 encoder layers for speed | `coderef/Pika-H3-Flex/` |
| MiniMax-H3 (release) | | | draft #91 Level-of-Token (mixed resolution, below dense quality by its author); issue #85 (unpruned brightness dips) | `coderef/MiniMax-H3/` |
| Comfy-Org templates | H3-Image graphs on stock `KSampler` euler/simple, six steps | | `TextGenerateV2` expanders on H3-Image | `coderef/workflow_templates/` |

## Node packs and community research

| source | the idea | what it touches here | read |
|---|---|---|---|
| ComfyUI-MAINodes | dials treat `MiniMaxH3ImageModel` like video, on the seam `MiniMaxH3Steer` uses; `H3RetimeAudio`; eye-colour shift through two SAM texts; still-region edit kept inside a SAM 3.1 word mask | control vectors, audio, masked lane | code (`coderef/ComfyUI-MAINodes/`) |
| KJNodes | an H3 prompt weight: per-token attention bias, where `(word:-1)` removes a concept; `[a|b:-2]` blending; a tiny VAE for H3-Image | prompting, control vectors, preview | commit diffs |
| Viggle Remake (Hugging Face) | `swap` and `edit` LoRAs on the base transformer, driven by one to four painted `<Picture N>` frames and a six-section prompt, with windows inside 90–345 | masked lane | card and file list |
| vllm-omni's character-replacement skill (rank-2 engine, community-style method) | reference-only swap with segment planning | masked lane | code |
| multi-swap, masked-latent and AV-mask-bridge packs | SAM 3 → ref2va swap → paste-back chains; community answers to stock `SetLatentNoiseMask` on AV latents | masked lane (what `MiniMaxH3MaskedSource` solves) | READMEs |
| Open-H3-Regenerate-2K-IC, H3Regenerate, FaceRegion | refine and upscale stages; per-token lower noise on faces | a possible refine stage | cards and READMEs |
| Continuity Studio, h3-editor, h3-long-video-research | visual memory across shots; editor-style timelines; latent carry against decode-reencode colour creep | long clips | READMEs |
| H3-Image Soft-Start, Conditioning Plus, Concept Sliders, Video-LoRA-for-Image | an empty picture for the first steps; neutral vision patches for t2i; concept text before the refiner; video LoRAs on the image model | image lane, control vectors. **No engine gives text to image any vision tokens** (2026-11-17, [`topics/qwen3vl_encoder.md`](topics/qwen3vl_encoder.md)), so the vision-patch idea runs the model on an input none of them use | READMEs |
| Sol-H3-VAE (`Efficient-Large-Model/Sol-H3-VAE`) and WanGP #2625 | a non-causal int8 video decoder reported faster and sharper than LightVAE | video decode | issue and listing |
| `roycho87/geiru_context_suite` (GPL-3.0) & `roycho87/NerdNotes` (CC BY-SA 4.0) | clean Clip-1 frame-0 keystone anchor + post-sample restore; exact within-slice mass-conserving attention re-splitting (`keystone <-> context`) preserving text/refs/audio shares; 1-clip perturbed-anchor block ablation isolating global appearance/luma blocks (`31–33, 36–44, 47`); host-RAM cropped Hybrid Windows | Shipped clean-room MIT equivalents in [`audio_freeze_song.py`](../../../audio_freeze_song.py) (`section_anchor`, `anchor_end_percent=0.33`, zero-mask context pin + `soft_empty_cache`, `anchor_weight`), [`attention.py`](../../../attention.py) (`lse_subset_balance_bhnd` on FlashAttention-2 / `sage-fork` `return_lse=True`, `attach_lse_reference_balance`, `_h3_audio_gain_applied` guard), and [`sol_attn_h3.py`](../../../sol_attn_h3.py) (`lse_balance` route); no GPL/BY-SA code copied (`4040e4dc`, `7b930ca1`, `73b74e55`, `6654678a`) | code + 4,629f 4-arm MP4 (`H`: [2026-11-16 audit](../research/2026-11-16_geiru_keystone_audit_and_roadmap.md), [Gate 1](../../../bench/results/2026-11-16_pika_anchor_and_audio_gain_probe.md), [Bucket 3](../../../bench/results/2026-11-16_song_section_anchor_ab.md), [Bucket 4](../../../bench/results/2026-11-16_lse_attention_balance_ab.md)) |
| `ckinpdx/ComfyUI-MMH3Tools` (MIT) | `Carry Attention` logit bias lowering carried-context weight and optionally raising `<Picture N>` reference weight (`carry_bias`, `reference_bias`) | Credited in clean-room [`section_anchor`](../../../audio_freeze_song.py) + superseded within-slice by exact mass-conserving `lse_subset_balance_bhnd` & `attach_lse_reference_balance` in [`attention.py`](../../../attention.py) | NerdNotes summary + formula check (`B`, [`bench/check_lse_balance.py`](../../../bench/check_lse_balance.py)) |
| `ethanfel/ComfyUI-MiniMaxH3-Context-Loop` (`Lumifel`) | tapered `context_noise` ramp fading from full noise at head to clean at join (`matched_noise_ratio`) | Shipped clean-room `context_profile="seam_taper"` in [`audio_freeze.py`](../../../audio_freeze.py#L436) & [`audio_freeze_song.py`](../../../audio_freeze_song.py) locking causal VAE `c - 2, c - 1` overlap (`7b930ca1`) | Geiru / MAINodes comparison (`B`, [Bucket 2](../../../bench/results/2026-11-16_seam_taper_and_audio_gain_guard.md)) |

## Distills (outside the owner's set, listed for awareness)

HyperFlow-H3-Flash, TurboH3, H3 AR few-step, FastVideo's NVFP4 Pika (Blackwell), CROWN HyperPDMD. The owner runs
no fast distill other than Pika.

## Papers (abstracts only)

SolarRenderer, AnyGameRenderer, VA-TDM, TileShare (a training-free speedup of the tiled decoder), VideoGen Model 1
(prompt expansion against formatting on a post-trained H3). The arXiv ids are in the sweep record.
