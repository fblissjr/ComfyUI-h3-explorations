# 17. PrunaAI `P-H3-Image`: Single-Frame T2I & Image Editing Prompting

> **Scope**: This section governs the **PrunaAI `P-H3-Image`** single-frame (`length=1`) text-to-image (`p_h3_image_t2i`) and multi-reference image-editing (`p_h3_image_edit`, 1–8 reference images) workflows in [`workflows/image/`](../../workflows/image/). It does **not** alter any rule in §1–§16 for H3 video (`t2va`, `i2va`, `fl2va`, `l2va`, `ref2va`).

---

## 17.1 Why `P-H3-Image` Uses a Separate Prompt Contract from H3 Video

`P-H3-Image` (`PrunaAI/P-H3-Image`, released 2026-11-05) was post-trained from the `MiniMax-H3-26B` (`fl2va`) DiT backbone for **single-frame (`t=1`), audio-free** image generation and editing, with five architectural changes that change what the text encoder and DiT expect:

1. **Distilled `Qwen3-VL-4B` Text Encoder + 5-Layer `TextConnector` (`p-h3-image_text_encoder_int8_convrot.safetensors`)**:
   - Instead of tapping a single layer (`hidden_states[-2]`, `dim=5120`) from `Qwen3-VL-32B-Instruct`, `PH3ImageTEModel` (`comfy/text_encoders/minimax.py:172-250`) extracts **5 intermediate hidden layers `[13, 19, 25, 31, 36]`** from `Qwen3-VL-4B` (`dim=2560` each), concatenates them along the channel axis (`5 * 2560 = 12800`), standardizes the `12800`-dim vector with learned `input_mean` and `input_std`, projects it in `float32` through a single bias-free linear layer `nn.Linear(12800, 5120, bias=False)` (`TextConnector.proj`, `comfy/text_encoders/minimax.py:110-128`), destandardizes to `output_mean` / `output_std` (`5120`), and replaces token 0 (`BOS`) with a learned `sink_embedding` (`5120`) into H3's `5120`-dim `condition_proj` + `TokenRefiner` space.
   - **Teacher-encoder compatibility (`h3_probe_p_h3_image_*_te32b_api.json`)**: Because `TextConnector` was distilled to match `Qwen3-VL-32B-Instruct`'s `5120`-dim layer-50 representation before `condition_proj`, the `P-H3-Image` DiT also accepts our stock `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` directly via `MiniMaxH3EncoderLoader`.
2. **Tokenizer Vocabulary (`PH3ImageTokenizer` vs. `MiniMaxH3Tokenizer`)**:
   - `PH3ImageTokenizer` (`comfy/text_encoders/minimax.py:254-258`) uses `Qwen3VLSDTokenizer` directly **without** registering `MINIMAX_EXTRA_TOKENS` (`<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>`, token IDs `151669..151675` in `comfy/text_encoders/minimax.py:33-35`).
   - Never use `<d>[Language] ...</d>`, `(S1)`, `overall_soundscape:`, `non_diegetic_music:`, or `[Shot N]` headers in a `P-H3-Image` prompt.
3. **Single-Frame DiT (`p-h3-image_int8_convrot.safetensors` or `h3/p-h3-image_lora_bf16.safetensors` on `unet_fl2va`) & Non-Uniform 8-Step Schedule**:
   - Runs at `t=1` (with two dummy audio latent frames discarded after `SamplerCustomAdvanced`), `KSamplerSelect("euler")`, `BasicGuider` (no CFG), `MiniMaxH3SigmaShift(shift_video=1.0, shift_audio=1.0)`, and `ManualSigmas` set to the non-uniform 8-step distillation grid `1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0` (`workflows/h3_config.py::PH3_IMAGE_MANUAL_SIGMAS`), spending 6 of its 8 steps in the structural high-noise window `[1.0 -> 0.81]`.
4. **Single-Frame Adapted H3 Image VAE (`p-h3-image_vae_bf16.safetensors` / `p-h3-image_vae_int8_convrot.safetensors`)**:
   - `PH3ImageVAE(MiniMaxH3VideoVAE)` (`comfy/ldm/minimax/vae.py:429-439, 1014-1033`) pairs MiniMax H3's 3D causal CNN encoder (`EncoderFCN3D`, using seeded diagonal-Gaussian sampling `PH3_IMAGE_ENCODE_SEED = 42` during `encode` and 24-channel `LATENTS_MEAN` / `LATENTS_STD` + ImageNet normalization) with `PH3ImageDecoder(ViT3DDecoder)` (`patch_size_t=1` and fixed temporal RoPE coordinate `PH3_IMAGE_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0` during `decode`, `16x` spatial compression, `24` latent channels).
5. **Dual-Stream Reference Conditioning (`MiniMaxH3PH3ImageConditioning`, `ph3_image_nodes.py`)**:
   - Creates both the conditioning (`["5", 0]`, with `_h3_span_masks` attached) and the single-frame latent (`["5", 1]`), at `2912x2912` (`~8.48 MP`, `PH3_IMAGE_CANVAS_8MP`) for T2I and `canvas="auto"` for Edit (which clamps `image_1` and the output canvas to `[4 MP, 8 MP]` at `image_1`'s aspect ratio, matching `coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:168-170`).
   - For editing (`1–8` reference images connected to `images.image_1` .. `images.image_8`), `MiniMaxH3PH3ImageConditioning` feeds each reference image simultaneously into:
     1. **Vision-Language Stream**: Each image is resized via `size_at_area` to `~1,000,000` pixels (multiples of `32`, keeping native size when already a multiple of `32` within 20%) and embedded at the `<|image_pad|>` token inside `<Picture i>: <|vision_start|><|image_pad|><|vision_end|>`.
     2. **VAE Reference-Latent Stream (`minimax_refs`)**: Under `canvas="auto"`, `image_1` is clamped to `[4 MP, 8 MP]` (`4*1024^2..8*1024^2`) and `image_2..image_8` are clamped to `[1 MP, 8 MP]` (`1024^2..8*1024^2`) via `size_in_area_range` (under `canvas="explicit"`, all `image_1..image_8` are clamped to `[1 MP, 8 MP]`, decoupled from the explicit output canvas), encoded with `PH3ImageVAE.encode` (`PH3_IMAGE_ENCODE_SEED = 42`), rounded to `torch.float16` in unnormalized latent space (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/encoders.py:64-65`, `coderef/MiniMax-H3/Ref2VA/video_vae/klvae.py:910`) before `(latents - mean) / std`, and injected as clean reference tokens (`minimax_refs`) into the DiT sequence.

---

## 17.2 Mode 1: Text-to-Image (`p_h3_image_t2i`)

Used by [`workflows/image/h3_p_h3_image_t2i_api.json`](../../workflows/image/h3_p_h3_image_t2i_api.json), [`workflows/image/h3_p_h3_image_lora_t2i_api.json`](../../workflows/image/h3_p_h3_image_lora_t2i_api.json), and [`workflows/image/h3_probe_p_h3_image_t2i_te32b_api.json`](../../workflows/image/h3_probe_p_h3_image_t2i_te32b_api.json).

### Format & Authoring Rules

- **Single cohesive prose paragraph (`80–200` words)**: Write one natural English paragraph (or 2–4 dense paragraphs for complex typography/infographics) with **no section headers** (`integrated_multimodal_description:`, `overall_soundscape:`, `non_diegetic_music:` are never used), **no Part One alignment line**, and **no `[Shot 1]` tag**.
- **Clause order**:
  1. **Framing, medium & primary subject**: Open with shot framing and medium (e.g., `"A medium three-quarter portrait of..."`, `"A wide architectural photograph of..."`), followed by concrete physical attributes, clothing, posture, and expression.
  2. **Environment & spatial layers**: Name foreground, midground, and background elements and how they relate spatially.
  3. **Lighting, color palette, lens & surface texture**: Specify key/fill/rim lighting direction and color temperature, depth of field, and tactile material textures (skin pores, fabric weave, brushed metal, wood grain).
- **In-scene visible text**: Put exact words to be rendered inside standard double quotes (`"..."`) and describe the typography, material, color, and surface placement (e.g., `hand-painted gold-leaf serif lettering reading "GRAND HOTEL" centered on the frosted glass transom`).

### Worked Example (`p_h3_image_t2i`, shipped in `h3_p_h3_image_t2i_api.json`)

```text
A medium three-quarter portrait of an elderly clockmaker seated at a worn walnut workbench inside a sunlit brass-and-oak workshop. He has silver stubble, deep-set hazel eyes behind round wire-rimmed spectacles, and wears a faded indigo linen apron over a rolled-sleeve cream cotton shirt. His weathered fingers hold a brass balance wheel and fine tweezers above an open pocket watch movement surrounded by miniature jeweler's screwdrivers, coiled mainsprings, and a glass bell jar. Warm late-afternoon directional sunlight streams through a paned workshop window on the left, casting crisp specular highlights across the polished gears and soft warm fill across the wood grain, shot on an 85mm prime lens at f/2.8 with shallow depth of field.
```

---

## 17.3 Mode 2: Multi-Reference Image Editing (`p_h3_image_edit`, 1–8 Images)

Used by [`workflows/image/h3_p_h3_image_edit_api.json`](../../workflows/image/h3_p_h3_image_edit_api.json), [`workflows/image/h3_p_h3_image_lora_edit_api.json`](../../workflows/image/h3_p_h3_image_lora_edit_api.json), and [`workflows/image/h3_probe_p_h3_image_edit_te32b_api.json`](../../workflows/image/h3_probe_p_h3_image_edit_te32b_api.json).

### How `MiniMaxH3PH3ImageConditioning` Wraps the Prompt

When `k` images (`images.image_1` .. `images.image_k`) are connected to `MiniMaxH3PH3ImageConditioning` (`ph3_image_nodes.py`), the node prepends `<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>...<Picture k>: <|vision_start|><|image_pad|><|vision_end|>` and applies `format_edit_prompt` (`coderef/diffusers/src/diffusers/modular_pipelines/p_h3_image/encoders.py:251-252`, `comfy_extras/nodes_minimax_h3.py:40-42, 724`):
- If `prompt` matches `^\s*(Edit <Picture \d+>|subject_definitions:)`, `MiniMaxH3PH3ImageConditioning` passes `prompt.strip()` verbatim.
- Otherwise, `MiniMaxH3PH3ImageConditioning` automatically prepends `"Edit <Picture 1>. "` to `prompt.strip()`.

### Format & Authoring Rules

- **Explicit `<Picture N>` tags (`1`-indexed)**: Refer to every connected input image by its exact tag (`<Picture 1>`, `<Picture 2>`, ..., up to `<Picture 8>`). `bench/check_prompt_guide_conformance.py::ph3_image_prompts_conform` verifies that an edit graph's prompt cites every wired `<Picture i>` and no unwired `<Picture m>`.
- **Single cohesive prose paragraph (`40–150` words)** stating three things in order:
  1. **Target transformation / synthesis**: What changes, combines, or transfers across `<Picture 1>`, `<Picture 2>`, etc.
  2. **Explicit retention anchor (`Keep ...`)**: What stays locked from each reference image (facial identity, bone structure, pose, camera angle, framing, clothing, or background geometry). Because `MiniMaxH3PH3ImageConditioning` injects clean VAE reference latents (`minimax_refs`) directly into the DiT sequence, naming what stays locked tells the DiT which spatial/identity features to preserve vs. synthesize fresh.
  3. **Physical integration**: How lighting direction, shadows, reflections, perspective, and surface textures unify across the edited result.

### Worked Example (`p_h3_image_edit`, shipped in `h3_p_h3_image_edit_api.json`)

```text
Place the man from <Picture 1> onto a stone observation terrace overlooking the alpine mountain peaks and valley from <Picture 2> at golden hour. Keep his exact facial features, expression, and hair from <Picture 1>, dress him in a charcoal wool field jacket over a heather-grey knit sweater, and match the warm low-angle sunset key light and crisp atmospheric perspective of <Picture 2>.
```

---

## 17.4 PrunaAI's `Qwen3.5-9B` Prompt Upsampler (`MiniMaxH3PH3ImagePromptUpsampler`)

PrunaAI ships an optional `Qwen3.5-9B` VLM rewriter (`p-h3-image_prompt_upsampler`, `9.1 GB` int8 `safetensors` in `<comfy>/models/text_encoders/p-h3-image_prompt_upsampler_int8_convrot.safetensors`, loaded via `CLIPLoader(type="qwen_image")`). In our canonical `workflows/image/` graphs, it is wired at node `25` (**`MiniMaxH3PH3ImagePromptUpsampler`**, `ph3_image_nodes.py`) with **`enabled = False` by default**.

### Why `enabled = False` Is the Default

1. **Zero VRAM / Zero Load Cost When Off**: `MiniMaxH3PH3ImagePromptUpsampler` marks `clip` as `lazy=True` in `define_schema()` and returns `[]` from `check_lazy_status()` when `enabled=False`. ComfyUI never executes node `24` (`CLIPLoader` for `p-h3-image_prompt_upsampler_int8_convrot.safetensors`), saving `9.1 GB` of disk/VRAM load and several seconds of autoregressive LLM generation while passing your authored `prompt` through untouched (`workflows/h3_config.py::Passthrough("prompt")`).
2. **Pre-Upsampled Prompts Are Deterministic and Faster**: Following the §17.2 / §17.3 rules (or copying the T2I / Edit system prompt blocks from [`docs/wiki/system_prompts/h3_p_h3_image_system_prompt.md`](../wiki/system_prompts/h3_p_h3_image_system_prompt.md)) produces dense, in-distribution prompts directly without running a second 9B LLM on every seed.
3. **Suboptimalities Fixed vs. PrunaAI's Reference Workflows**:
   - PrunaAI's T2I workflow wired `ChatTemplate` (a custom node not in ComfyUI core) or core's `TextGenerateLTX2Prompt` (which injects Gemma `<start_of_turn>` tags and LTX-2's video system prompt), and its Edit workflow wired `TextEncodeQwenImageEditPlus` (which accepts at most 3 images, downscales them to `384x384`, injects Qwen-Image-Edit's system prompt, and outputs `CONDITIONING` instead of calling `clip.generate`).

### How `MiniMaxH3PH3ImagePromptUpsampler` Works When `enabled = True`

When you flip `enabled = True` on node `25` (`MiniMaxH3PH3ImagePromptUpsampler`) to expand a short request:

1. **Exact Upstream ChatML Format (`ph3_image_nodes.py::build_upsampler_chat`)**:
   - **No reference images connected (T2I, `max_new_tokens = 1024`)**: Computes the closest aspect-ratio bucket (`1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`) from `width` and `height` and formats:
     ```text
     <|im_start|>user
     Aspect ratio: {aspect_ratio}
     Request: {prompt}<|im_end|>
     <|im_start|>assistant
     <think>
     ```
   - **1–8 reference images connected (Edit, `max_new_tokens = 1536`)**: Resizes each reference image at its native aspect ratio to `~1 MP` (`1,000,000` pixels, multiples of `32`, Lanczos, no white padding) via `ph3_image_nodes.py::_resize_upsampler_image` and formats:
     ```text
     <|im_start|>user
     <Picture 1>: <|vision_start|><|image_pad|><|vision_end|>[<Picture 2>: ... ]Edit <Picture 1>. {prompt}<|im_end|>
     <|im_start|>assistant
     <think>
     ```
2. **Validation of `<think>...</think>` and Rewritten Output**:
   - Runs deterministic greedy decoding (`do_sample=False, mtp=True`) via `clip.generate`, splits on `</think>` to return `(rewritten, thinking)`, and verifies that `</think>` closed within `max_new_tokens` and (for edits) that `rewritten` conforms to `ph3_image_nodes.py::is_edit_prompt`.
