# PrunaAI `P-H3-Image`: Single-Frame T2I & Multi-Reference Image Editing

last updated: 2026-11-07 (initial page: how PrunaAI built `P-H3-Image` from `MiniMax-H3-26B`, exact architectural and tensor differences against stock H3, what was wrong in PrunaAI's reference ComfyUI workflows, and how our 6 workflows in `workflows/image/` and `MiniMaxH3PH3ImagePromptUpsampler` work)

This page is the technical reference in the wiki for PrunaAI's `P-H3-Image` (`PrunaAI/P-H3-Image`, released 2026-11-05) and `PrunaAI/P-H3-Image-Comfy` in this repo.
- **Prompting rules, `<Picture i>` indexing, and worked examples**: [`docs/prompting/17_p_h3_image_prompting.md`](../prompting/17_p_h3_image_prompting.md) and [`docs/wiki/system_prompts/h3_system_prompt.md`](system_prompts/h3_system_prompt.md) (`MODE: p_h3_image_t2i` and `MODE: p_h3_image_edit`).
- **Workflow layout and historical 2026-08 single-frame context**: [`docs/h3_image_editing.md`](../h3_image_editing.md).

---

## 1. How PrunaAI Built `P-H3-Image` from `MiniMax-H3-26B`

`P-H3-Image` adapts the 26B `MiniMax-H3` (`fl2va`) audio-video diffusion transformer for **single-frame (`t=1`), audio-free** image generation (`~8.48 MP`, `2912x2912` default) and multi-reference image editing (`1..8` reference images). Across the checkpoint files and `diffusers` `src/diffusers/modular_pipelines/p_h3_image/`, five components differ from stock MiniMax-H3:

| Component | Stock MiniMax-H3 Video (`fl2va` / `ref2va`) | PrunaAI `P-H3-Image` (`workflows/image/`) | ComfyUI Core & Pack Code |
|---|---|---|---|
| **DiT Backbone & LoRA** | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (`20–50` steps or 6-step Pika `lcm`) | `p-h3-image_int8_convrot.safetensors` (`24 GB` merged int8 ConvRot) **or** `h3/p-h3-image_lora_bf16.safetensors` (`3.5 GB`, `rank=128` across all 50 DiT blocks + `token_refiner`) applied onto `unet_fl2va` via [`MiniMaxH3LoRABranch`](../../lora_branch.py) | `comfy/ldm/minimax/model.py`; [`lora_branch.py`](../../lora_branch.py) |
| **Step & Sigma Schedule** | `MiniMaxH3SigmaShift(shift_video=12.0, shift_audio=3.0)` + `BasicScheduler("simple")` | `MiniMaxH3SigmaShift(shift_video=1.0, shift_audio=1.0)` + `ManualSigmas` set to the non-uniform 8-step distillation grid `1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0` (`PH3_IMAGE_MANUAL_SIGMAS`), `KSamplerSelect("euler")`, `BasicGuider` (`cfg=1.0`). 6 of the 8 steps sit in `[1.0 -> 0.81]` | [`workflows/h3_config.py`](../../workflows/h3_config.py) (`PH3_IMAGE_MANUAL_SIGMAS`, `PH3_IMAGE_SHIFT`) |
| **Text Encoder (`5120`-dim)** | `Qwen3-VL-32B-Instruct` (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, layer 50 / penultimate `hidden_states[-2]`, `dim=5120`) | Distilled `Qwen3-VL-4B` (`p-h3-image_text_encoder_int8_convrot.safetensors`, `4.7 GB`, loaded via `CLIPLoader(type="minimax")`) + FP32 `TextConnector`: concatenates 5 hidden layers `[13, 19, 25, 31, 36]` (`5 * 2560 = 12,800`), standardizes with learned `input_mean` / `input_std`, projects via bias-free `nn.Linear(12800, 5120, bias=False)` (`TextConnector.proj`), destandardizes to `output_mean` / `output_std` (`5120`), and replaces token 0 (`BOS`) with learned `sink_embedding` (`norm=15810.24`). **Teacher-compatible**: also accepts our 32B `MiniMaxH3EncoderLoader` directly (`h3_probe_p_h3_image_*_te32b_api.json`) | `comfy/text_encoders/minimax.py:110-128, 172-250` (`TextConnector`, `PH3ImageTEModel`) |
| **Tokenizer** | `MiniMaxH3Tokenizer` (`Qwen3VLSDTokenizer` + 7 `MINIMAX_EXTRA_TOKENS` `151669..151675`) | `PH3ImageTokenizer` (`Qwen3VLSDTokenizer` **without** `MINIMAX_EXTRA_TOKENS`; `<d>`, `</d>`, `<|cutoff|>`, etc. are not registered) | `comfy/text_encoders/minimax.py:33-35, 254-258` |
| **Single-Frame VAE** | `MiniMaxH3VideoVAE` (`minimax_h3_video_vae_bf16.safetensors`) | `PH3ImageVAE(MiniMaxH3VideoVAE)` (`p-h3-image_vae_bf16.safetensors` / `p-h3-image_vae_int8_convrot.safetensors`): H3's 3D causal CNN encoder (`EncoderFCN3D`) with seeded diagonal-Gaussian sampling (`PH3_IMAGE_ENCODE_SEED = 42`), 24-channel `LATENTS_MEAN` / `LATENTS_STD` + ImageNet normalization, and `PH3ImageDecoder(ViT3DDecoder)` (`patch_size_t=1` and fixed temporal RoPE coordinate `PH3_IMAGE_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0`) | `comfy/ldm/minimax/vae.py:429-439, 1014-1033` |
| **Conditioning & Reference Stream** | `MiniMaxH3Conditioning` / `MiniMaxH3ReferenceConditioning` | `PH3ImageConditioning`: creates both `conditioning` and the single-frame AV latent (`_av_latent(latent_width, latent_height, 1, 2)`). On T2I graphs, `width=2912, height=2912` (`~8.48 MP`, `33,124` patch tokens). On Edit graphs (`1..8` images), `width=0, height=0` so `latent_width = refs[0]["latent_w"] * 16` and `latent_height = refs[0]["latent_h"] * 16` preserve `image_1`'s aspect ratio while scaling each reference image to `<= 1 MP` for Qwen3-VL vision tokens and `<= area` for `PH3ImageVAE.encode` (`minimax_refs`) | `comfy_extras/nodes_minimax_h3.py:668-732` |
| **Prompt Upsampler** | External authoring with [`h3_system_prompt.md`](system_prompts/h3_system_prompt.md) | Optional fine-tuned `Qwen3.5-9B` (`p-h3-image_prompt_upsampler_int8_convrot.safetensors`, `9.1 GB`, loaded via `CLIPLoader(type="qwen_image")`), wrapped by [`MiniMaxH3PH3ImagePromptUpsampler`](../../ph3_image_nodes.py) (`enabled=False` by default, `lazy=True` on `clip`) | [`ph3_image_nodes.py`](../../ph3_image_nodes.py) |

---

## 2. Suboptimalities Fixed vs. PrunaAI's Reference ComfyUI Workflows

PrunaAI's upstream example workflows (`PrunaAI/P-H3-Image-Comfy`) had four defects that degraded quality or failed to run cleanly on ComfyUI core:

1. **Wrong Sigma Schedule (`ModelSamplingSD3(shift=1.0)` + `BasicScheduler("simple", 8)`)**:
   - Upstream's example JSONs wired `ModelSamplingSD3(shift=1.0)` + `BasicScheduler(scheduler="simple", steps=8)`, which produces a uniform linear schedule `[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]`.
   - `PrunaAI/P-H3-Image`'s actual `modular_model_index.json` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:837-838`) uses the non-uniform 8-step distillation schedule `1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0`, where 6 of the 8 steps are concentrated in the high-noise structural formation window `[1.0 -> 0.81]`.
   - Our workflows wire `MiniMaxH3SigmaShift(shift_video=1.0, shift_audio=1.0)` + `ManualSigmas(PH3_IMAGE_MANUAL_SIGMAS)`, guarded by [`bench/check_distill_settings.py`](../../bench/check_distill_settings.py) and [`bench/check_distill_grid.py`](../../bench/check_distill_grid.py).
2. **Broken T2I & Edit Prompt Upsampler Node Chains**:
   - Upstream's T2I workflow wired `ChatTemplate` (a custom node absent from ComfyUI core) or core's `TextGenerateLTX2Prompt` (which wraps prompts in Gemma `<start_of_turn>` tags and injects LTX-2's video system prompt).
   - Upstream's Edit workflow wired `TextEncodeQwenImageEditPlus` (`comfy_extras/nodes_qwen.py`), which accepts at most 3 images, downscales them to `384x384` (`147,456` px vs. upstream's `1,000,000` px), injects Qwen-Image-Edit's system prompt, and outputs `CONDITIONING` rather than calling `clip.generate`.
   - Our [`MiniMaxH3PH3ImagePromptUpsampler`](../../ph3_image_nodes.py) (node `25`) implements the exact upstream `prompt_upsampler.py` contract:
     - `enabled=False` by default with `lazy=True` on `clip` (`check_lazy_status` returns `[]` when `enabled=False`), so the `9.1 GB` `Qwen3.5-9B` checkpoint is never loaded unless opted into.
     - When `enabled=True`, verifies the connected `clip` is not the MiniMax text encoder, formats the exact ChatML user-only turn + `<think>\n` prefill (`1024` tokens with log-space aspect-ratio bucketing for T2I; `1536` tokens with `1..8` reference images resized at native aspect ratio to `~1 MP` at multiples of `32` via `size_at_area` for edits), checks `QWEN35_IM_END_ID` (`248046`) before `clip.decode` (which strips special tokens by default), verifies `decoded.count("</think>") == 1`, and validates edit output against `is_edit_prompt`.
3. **Dense Attention Floor (`ModelAttentionBackend(attention="comfy_kitchen_int8")`)**:
   - Upstream's workflows omitted `ModelAttentionBackend`. Our 6 workflows wire `ModelAttentionBackend(attention="comfy_kitchen_int8")` (node `58`) for rotated int8 dense attention on single-frame graphs (Sol-Attn is a 3D spatiotemporal video kernel and is exempt on single-frame graphs via `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`).
4. **Runtime `MiniMaxH3LoRABranch` (`rank=128` bf16) & 32B Teacher Encoder Probes**:
   - Applying `h3/p-h3-image_lora_bf16.safetensors` (`rank=128`) onto `minimax_h3_fl2va_pruned_int8_convrot.safetensors` via [`MiniMaxH3LoRABranch`](../../lora_branch.py) preserves full bf16 LoRA precision without int8 merge requantization loss ([`docs/research/merge_requantisation.md`](../research/merge_requantisation.md)).
   - Because `TextConnector` was distilled to match `Qwen3-VL-32B-Instruct`'s `5120`-dim layer-50 representation, `h3_probe_p_h3_image_t2i_te32b_api.json` and `h3_probe_p_h3_image_edit_te32b_api.json` wire our stock 32B `MiniMaxH3EncoderLoader` directly into `PH3ImageConditioning`.

---

## 3. Canonical Workflows & Code Map

| File / Constant | Role | Guarded By |
|---|---|---|
| [`ph3_image_nodes.py`](../../ph3_image_nodes.py) | `MiniMaxH3PH3ImagePromptUpsampler` node + `aspect_ratio_label`, `size_at_area`, `is_edit_prompt`, `build_upsampler_chat` | [`bench/check_schema_defaults.py`](../../bench/check_schema_defaults.py), [`bench/check_node_ids.py`](../../bench/check_node_ids.py), [`bench/check_literal_widgets.py`](../../bench/check_literal_widgets.py) |
| [`workflows/h3_config.py`](../../workflows/h3_config.py) (`PH3_IMAGE_*`, `SINGLE_FRAME_DIR`) | Single source of truth for model filenames, 8-step sigmas, shift `(1.0, 1.0)`, canvas sizes, and upsampler token budgets | [`bench/check_model_files.py`](../../bench/check_model_files.py), [`bench/check_distill_settings.py`](../../bench/check_distill_settings.py), [`bench/check_distill_grid.py`](../../bench/check_distill_grid.py) |
| [`workflows/build_workflows.py`](../../workflows/build_workflows.py) (`build_ph3_image_api`) | Generates all 6 workflows in `workflows/image/` | [`bench/check_widget_deviations.py`](../../bench/check_widget_deviations.py), [`bench/check_attention_defaults.py`](../../bench/check_attention_defaults.py) |
| `prompt_bank/p_h3_image/t2i_clockmaker.txt`, `prompt_bank/p_h3_image/edit_alpine_terrace.txt` | House prompts for the T2I and Edit workflows, loaded via [`workflows/prompts.py`](../../workflows/prompts.py) (`text`) | [`bench/check_prompt_guide_conformance.py`](../../bench/check_prompt_guide_conformance.py) (`ph3_image_prompts_conform`) |
| [`docs/prompting/17_p_h3_image_prompting.md`](../prompting/17_p_h3_image_prompting.md) & [`docs/wiki/system_prompts/h3_system_prompt.md`](system_prompts/h3_system_prompt.md) | Prompting contract and LLM system prompt modes (`p_h3_image_t2i`, `p_h3_image_edit`) | [`bench/check_prompt_docs_sync.py`](../../bench/check_prompt_docs_sync.py), [`bench/check_doc_links.py`](../../bench/check_doc_links.py) |
