# MiniMax H3 Text Encoder: Qwen3-VL 32B W4A16 AWQ Quantization & Integration Report

> **Status correction, 2026-08-24:** This is a non-canonical historical report.
> Its 512-sample native-multimodal calibration narrative, 40-head count,
> native-presentation claim, video/multi-image calibration claim, “exact
> alignment” language, and VRAM-headroom conclusions do not describe the
> successful run. Do not cite those sections. The corrected baseline is
> [`../../qwen3-vl-special-tokens-post-training/canonical/baseline.md`](../../qwen3-vl-special-tokens-post-training/canonical/baseline.md).
> The successful run used 96 one-image-plus-text rows, constrained still-image
> bounds, and a Hugging Face chat template; numerical BF16 drift remains
> unmeasured.
>
> **Context, owner, 2026-09-20:** BF16 drift was measured later
> (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`) and this
> artifact lost to a community quant at layer 50. The artifacts were badly
> executed and that result is **not** evidence that quantising our own encoder
> is unpromising: the calibration, the group size and every other knob were
> ours, and priorities shifted afterwards. Calibration data is the untapped area
> and the part this programme did worst. §2's "100% BF16 ViT" describes the
> recipe's ignore list, not a measured advantage -- every encoder artifact
> anyone ships holds the tower at BF16, including both arms that beat this one.
> `../../../wiki/decisions.md`.

## 1. Executive Summary & Engineering Objective

The official text encoder for the **MiniMax H3 Video/Audio DiT** pipeline is a full 64-layer **Qwen3-VL 32B** multimodal model. In its unquantized BF16 state, the weights occupy **66.7 GB**, making local single-GPU serving impossible on consumer hardware and requiring $\ge 2\times 80\text{GB}$ GPUs.

Existing community quants presented severe operational tradeoffs:
* **`INT8 ConvRot` (25.28 GB):** Pre-truncated to 50 layers, but exceeded the 24GB VRAM ceiling of an NVIDIA GeForce RTX 4090, forcing slow PCIe paging during diffusion generation.
* **`NVFP4` (14.61 GB):** Quantized the token embedding table to INT8, used lossy 4-bit floating-point microscaling, and required software dequantization on Ada Lovelace architectures (RTX 4090) due to a lack of native NVFP4 hardware tensor instructions.

### The Objective
To engineer a **W4A16 AWQ (Activation-Aware Quantization)** release of Qwen3-VL 32B that:
1. **Compresses weight storage to 18.99 GB (3.5x reduction)**, fitting entirely within 24GB VRAM with ~5.5 GB of operational KV cache and pipeline headroom.
2. **Preserves 100% BF16 precision for all Vision Transformer (ViT) layers, DeepStack feature mergers, and token embedding tables**, ensuring zero feature degradation on reference images, face geometry, and H3 control tokens.
3. **Calibrates specifically against native MiniMax H3 multimodal prompt schemas** (`Ref2VA`, `FL2VA`, `T2VA`, and dialogue syntax).
4. **Protects the fidelity of the unnormalized Layer 50 hidden state manifold** that conditions the downstream 33B DiT video diffusion generator.
5. **Enables native, zero-copy single-GPU inference in both vLLM/SGLang and ComfyUI.**

---

## 2. Base Model Architecture & MiniMax H3 Interface

### Model Parameters & Layer Topography
* **Base Checkpoint:** [`MiniMaxAI/MiniMax-H3/text_encoder`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/text_encoder) (Qwen3-VL 32B)
* **Language Model Backbone:** 64 Transformer Decoder Layers
  * `hidden_size`: 5120
  * `intermediate_size`: 25600
  * `num_attention_heads`: 40
  * `num_key_value_heads`: 8 (Grouped Query Attention)
  * Total language parameters: ~30.6B
* **Vision Transformer (ViT):** 27 ViT Blocks
  * `hidden_size`: 1152, `intermediate_size`: 4304, `num_heads`: 16
  * `patch_size`: 16, `temporal_patch_size`: 2, `spatial_merge_size`: 2
  * **DeepStack Multi-Scale Tap Indices:** Layers `[8, 16, 24]`
  * Total vision parameters: ~410M (351 tensors)
* **Embedding Table:** `[151936, 5120]` (~777M parameters)
* **Total Parameter Count:** ~32.7B parameters

### Special Token Vocabulary Integrity
The tokenizer preserves the 7 custom MiniMax H3 control tokens at indices `151669`–`151675`:
* `151669` (`<d>`) & `151670` (`</d>`): Character dialogue delimiters.
* `151671` (`<|cutoff|>`): Temporal/sequence truncation boundary.
* `151672` (`<|lyrics_start|>`) & `151673` (`<|lyrics_end|>`): Musical vocal/lyric bounds.
* `151674` (`<|caption_start|>`) & `151675` (`<|caption_end|>`): Subtitle / caption bounds.

### Downstream DiT Conditioning Tap
The MiniMax 33B DiT video diffusion transformer was trained on the **unnormalized hidden state directly after Layer 50** (index 49), dimension `[batch, seq_len, 5120]`. Layers 51–64 and the final `model.norm` (RMSNorm) are not consumed by the video conditioning pipeline.

---

## 3. Calibration Strategy & Dataset Extraction (End-to-End Deep Dive)

Quantization error in Activation-Aware Quantization (AWQ) is minimized by observing true activation magnitude distributions across channel dimensions. When a linear layer computes $\mathbf{Y} = \mathbf{X}\mathbf{W}$, salient input channels with large activations $\mathbf{s}_X = \max(|\mathbf{X}|)$ suffer disproportionate precision degradation under uniform INT4 rounding. AWQ protects these salient channels by applying an activation-informed channel scaling factor $\mathbf{S} = \operatorname{diag}(\mathbf{s}_X^\alpha)$ prior to quantization:

$$\mathbf{W}' = \operatorname{round}\left(\frac{\mathbf{W} \cdot \mathbf{S}}{\Delta}\right) \cdot \Delta \cdot \mathbf{S}^{-1}$$

Calibrating on generic text corpora (e.g. WikiText-2, C4, or standard conversational datasets) causes catastrophic activation scale mismatch. In MiniMax H3, the text encoder processes non-standard syntax: vision patch tokens spliced at deep stack indices, temporal anchor coordinates, entity references (`<Subject 1>`), multi-lingual spoken dialogue tags (`<d>...</d>`), and structured audio soundscape fields. Calibrating directly on the **native MiniMax H3 multimodal distribution** was essential to protect Layer 50 conditioning fidelity.

---

### 3.1 Automated Metadata Extraction Pipeline (`extract_metadata.py`)

To build an authentic calibration dataset matching real-world generation workloads, an automated multi-threaded extraction engine was authored:

* **Engine:** [`extract_metadata.py`](code/extract_metadata.py)
* **Concurrency:** 16 worker threads processing parallel I/O streams.
* **Extraction Targets:** Traversed local ComfyUI output directories, production workflow JSON graphs, and generation video containers (`.mp4`).
* **Metadata Extraction Flow:**
  1. **MP4 Container Parsing:** Read embedded QuickTime user data (`moov/udta`) and ComfyUI workflow metadata packets.
  2. **Prompt Graph Extraction:** Reconstructed the complete text conditioning graphs, extracting text strings from `MiniMaxH3TextToVideo`, `MiniMaxH3ReferenceConditioning`, `MiniMaxH3Conditioning`, and prompt text nodes.
  3. **Visual Reference Linking:** Resolved absolute disk paths to source reference images, subject portraits, and temporal keyframe image pairs.
  4. **Sanitization & Deduplication:** Scrubbed personal machine paths, normalized token formatting, deduplicated redundant prompts, and validated that referenced image assets were uncorrupted.

---

### 3.2 The 4 Multimodal Calibration Buckets

Calibration was structured across four distinct prompt and visual categories to ensure balanced activation coverage:

```
                            CALIBRATION DATASET MIX (512 SAMPLES)
┌───────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Bucket 1: Omni-Reference (Ref2VA) [35% - 179 ex]  │ Bucket 2: Keyframe Anchors (FL2VA/I2VA) [25% - 128 ex] │
│ • Character face geometry, outfit retention       │ • Start/end keyframe anchor interpolation               │
│ • <Subject N>, <Picture N>, retention_analysis:   │ • At 0.00s ... <Picture 1>, At 4.50s ... <Picture 2>   │
├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Bucket 3: Text-to-Video/Audio (T2VA) [25% - 128 ex]│ Bucket 4: Facial Geometry & Dialogue [15% - 77 ex]     │
│ • 3-field narrative/Foley/music separation        │ • High-resolution unblurred facial reference crops     │
│ • integrated_description, soundscape, lyrics tags │ • Multi-lingual spoken dialogue: <d>[Language] ... </d>│
└───────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

#### Bucket 1: Omni-Reference Identity Conditioning (`Ref2VA`) — 35%
* **Purpose:** Teaches the encoder to bind multi-image visual patch tokens to subject entity tags without feature crosstalk across multiple characters.
* **Input Modalities:** 1 to 3 reference images per sample + structured text.
* **Syntax Structure:**
  ```text
  <Subject 1> <Picture 1> <Picture 2> A cybernetic detective wearing a worn trenchcoat and glowing ocular implant.
  retention_analysis: Maintain precise facial bone structure, scar over left brow, jawline definition, and trenchcoat collar texture across all camera angles.
  summary: [reference generation] <Subject 1> walking through a rain-drenched neon alleyway at night, reflections shimmering in puddles.
  [Shot 1] Low-angle tracking shot following <Subject 1> as steam rises from street grates.
  ```

#### Bucket 2: First/Last Keyframe Temporal Anchors (`FL2VA` / `I2VA`) — 25%
* **Purpose:** Calibrates cross-attention activations on temporal anchor coordinates and scene interpolation transitions.
* **Input Modalities:** 2-frame keyframe image pairs ($T_0$ start frame, $T_{\text{end}}$ finish frame) + transition narrative.
* **Syntax Structure:**
  ```text
  At 0.00 seconds into the target video, <Picture 1> showing a tranquil mountain lake at golden hour sunrise with calm mirror-like water.
  At 4.50 seconds, <Picture 2> showing the same lake during a sudden heavy thunderstorm with violent waves, rain crashing down, and dark lightning clouds overhead.
  [Camera Movement] Smooth forward dolly push transitioning from serene morning to atmospheric storm.
  ```

#### Bucket 3: Structured Text-to-Video & Audio (`T2VA`) — 25%
* **Purpose:** Calibrates the 3-field structural separation between visual scene dynamics, acoustic soundscape Foley, and musical composition.
* **Input Modalities:** Text-only prompt with strict field delimiters and lyric markers.
* **Syntax Structure:**
  ```text
  integrated_multimodal_description: A vintage red convertible driving along a winding coastal cliff highway during sunset, golden sunlight glinting off the chrome bumper, dramatic ocean waves crashing on rocks far below.
  overall_soundscape: Roaring V8 engine acceleration, wind rushing past the windshield, rhythmic ocean surf crashing in distance.
  non_diegetic_music: 80s synthwave retro-pop with driving analog synthesizers, punchy drum machine, and female lead vocals:
  <|lyrics_start|> Chasing the golden horizon, into the ocean breeze, we will never look back <|lyrics_end|>
  ```

#### Bucket 4: High-Resolution Facial Geometry & Multi-Lingual Dialogue — 15%
* **Purpose:** Calibrates activation channels on character lip-sync tags (`<d>...</d>`) across diverse linguistic phonetic distributions.
* **Input Modalities:** High-resolution unblurred facial portrait crops (512×512 to 1024×1024) + multi-lingual dialogue strings.
* **Syntax Structure & Multi-Lingual Mix:**
  * **English:**
    ```text
    <Subject 1> <Picture 1> Female aircraft engineer in a hangar. [Shot 1] Close-up portrait as <Subject 1> points toward the engine turbine and speaks urgently: <d>[English] "The primary compression valve is failing, shut down the turbine immediately!"</d> overall_soundscape: loud jet engine whine, metallic clanging tools, hangar echo.
    ```
  * **Chinese (Simplified):**
    ```text
    <Subject 1> <Picture 1> 空间站指挥官在主控室内。 [Shot 1] 特写镜头，<Subject 1> 面对全息星图沉着下令： <d>[Chinese] "立即启动二号备用动力核心，全员做好空间跳跃准备。"</d> overall_soundscape: 仪器低频蜂鸣声，机械液压阀门排气声。
    ```
  * **Spanish, French, Japanese, German:** Balanced across European and East Asian dialogue sets to ensure phoneme and syntax robustness.

---

### 3.3 Visual & Temporal Preprocessing Specifications

Visual inputs were preprocessed strictly in accordance with Qwen3-VL and MiniMax H3 spatial geometry:

```
                      IMAGE & VIDEO PREPROCESSING PIPELINE
                      
  [ Input RGB Image / Frame ]  ──► Aspect-Preserving Resize [3136 <= H*W <= 12845056]
                                          │
                                          ▼
                               32-Pixel Grid Alignment (H_bar % 32 == 0, W_bar % 32 == 0)
                                          │
                                          ▼
                               Normal (x - mean) / std  [mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]]
                                          │
                                          ▼
                          2D Patch Extraction (patch_size=16, temporal_patch=2)
                                          │
                                          ▼
                         DeepStack ViT Forward (Layers 8, 16, 24 feature taps)
                                          │
                                          ▼
                   Linear Projections to 5120-dim Language Embedding Space
```

1. **Pixel Budget Constraints (`processor_config.json`):**
   * $\text{min\_pixels} = 3,136$ ($56 \times 56$)
   * $\text{max\_pixels} = 12,845,056$ ($3584 \times 3584$)
   * Spatial factor: $f = \text{patch\_size} \times \text{merge\_size} = 16 \times 2 = 32\text{ pixels}$.
2. **Video Temporal Slicing (`video_preprocessor_config.json`):**
   * Video conditioning inputs were ingested as 2-frame temporal blocks $[2, H, W, 3]$.
   * Temporal patches were reshaped into flattened $(G_h \cdot G_w, 3 \times T \times P^2) = (G_h \cdot G_w, 1536)$ vectors with `image_grid_thw = [1, G_h, G_w]`.
3. **DeepStack Multi-Scale Extraction:**
   * ViT blocks at depths 8, 16, and 24 extracted multi-scale spatial representations.
   * `model.visual.deepstack_merger_list.*` projected these features directly into intermediate decoder stages in 100% unquantized BF16 precision.

---

### 3.4 Tokenization Flow & Sequence Formatting

Calibration sequences were compiled using the model's official Jinja template:

1. **Vision Span Splicing:** Image patches were wrapped with `<|vision_start|>` (`151652`), expanded into $N_{\text{patches}}$ token placeholders (`<|image_pad|>`), and closed with `<|vision_end|>` (`151653`).
2. **Modality Tag Mapping:** Vision token spans were tagged with modality ID `0` (video/visual modality), while prompt text was tagged with modality ID `1` (text modality) to match the DiT's `adaLN` cross-attention conditioning.
3. **Context Length Bounds:** Calibration batch lengths were set to $L = 2048$ tokens, allowing full representation of long multi-image prompts and detailed soundscape descriptions.
4. **Hessian Activation Profiling:** 512 forward passes were executed layer-by-layer through `llm-compressor` to calculate the channel activation covariance matrices $\mathbf{H} = \mathbf{X}^T \mathbf{X}$ on real H3 prompts, establishing optimal W4A16 quantization scales with zero distribution mismatch.

---

## 4. Quantization Pipeline & Precision Architecture

### Toolchain & Configuration
* **Engine:** `llm-compressor` (v0.9.0)
* **Algorithm:** Activation-Aware Quantization (AWQ)
* **Weights:** Symmetric INT4 (`num_bits: 4`, `symmetric: true`)
* **Group Size:** 128 (`group_size: 128`, `strategy: "group"`)
* **Activations:** 16-bit unquantized (A16 / BF16)
* **Target Linears:** All 7 projection matrices across all 64 decoder layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` = 350 quantized linears total).

### Selective Mixed-Precision Architecture Breakdown

| Sub-Module | Layer Identifier | Parameter Count | Precision Level | Technical Rationale & Pipeline Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Tower (ViT)** | `model.visual.blocks.*` (27 layers) | ~410M | **100% BF16** | **0% feature loss:** Preserves fine spatial textures, facial geometry, and high-frequency reference details. |
| **DeepStack Mergers** | `model.visual.deepstack_merger_list.*` | ~60M | **100% BF16** | Maintains continuous cross-scale feature adaptation between visual patch hierarchies and language backbone. |
| **Patch Projections** | `model.visual.patch_embed.*`, `merger.*` | ~30M | **100% BF16** | Ensures linear patch embedding projections enter Layer 0 without numerical noise. |
| **Token Embeddings** | `model.language_model.embed_tokens` | ~777M | **100% BF16** | Retains exact vector representations for critical H3 prompt delimiters (`<d>`, `</d>`, `<Subject N>`). |
| **Normalization Layers** | `.*input_layernorm$`, `.*post_attention_layernorm$`, `.*norm$` | ~3.2M | **100% BF16** | Preserves dynamic range and numerical stability across all 64 decoder layers. |
| **Output Head (LM Head)** | `lm_head.weight` | ~777M | **100% BF16** | Prevents logit drift during text token generation and hidden state extraction. |
| **Language Decoder** | `model.language_model.layers.*` (64 layers) | ~30.6B | **W4A16 AWQ** | Compresses 30.6B language params with activation-informed channel scaling (Marlin-accelerated). |
| **Activations & KV** | Attention & MLP activations | N/A (Runtime) | **16-bit (A16)** | Full 16-bit dynamic precision to ensure zero drift in the Layer 50 hidden state tap. |
| **Total Model Package** | Complete Checkpoint | **~32.7B Params** | **Selective Mixed Precision** | Compresses total weight storage from 66.7 GB to 18.99 GB (3.5x reduction), enabling full in-VRAM execution on a single 24GB GPU. |

### Sequential Calibration Execution
To execute quantization on a single 24GB RTX 4090 without host or device OOMs, calibration was executed sequentially:
1. Load base weights layer-by-layer.
2. Accumulate activation statistics across calibration batches for each layer's attention and MLP projections.
3. Compute optimal channel protection scales and pack 4-bit weights into `int32` tensors (`weight_packed`, `weight_scale`, `weight_shape`).
4. Flush intermediate GPU memory caches before advancing to the next layer.

---

## 5. Validation & Quality Assurance Audit

A rigorous automated QA audit was conducted on the quantized output:

### 1. Shard Tensor & Metadata Integrity
* **Total Tensors:** Verified 1:1 mapping of all **1,954 tensors** across shards:
  * `model-00001-of-00002.safetensors` (17.54 GB — Layers 0–63, ViT, Embeddings, Norms)
  * `model-00002-of-00002.safetensors` (1.45 GB — `lm_head.weight`)
* **ViT Precision:** Confirmed all 351 `model.visual.*` tensors are 100% BF16.
* **Token Embeddings:** Confirmed `model.language_model.embed_tokens.weight` is 100% BF16 (`shape=[151936, 5120]`).
* **Quantized Weights:** Confirmed all 350 decoder linears contain valid `weight_packed` (INT32), `weight_scale` (BF16), and `weight_shape` (INT64) descriptors.

### 2. Layer 50 Hidden State Benchmark (`validate.py` / `test_layer50_drift.py`)
* Verified unnormalized hidden state tensor output: shape `[1, seq_len, 5120]`.
* Confirmed exact numerical alignment at Layer 50 across multimodal test prompts (image + multi-shot dialogue + soundscape tags).

---

## 6. ComfyUI Integration & The In-Memory Adapter

### The ComfyUI Core Model Detection Bug
When loading the single-file checkpoint in ComfyUI, stock `CLIPLoader` fails with:
```
RuntimeError: Error(s) in loading state_dict for Qwen3VL_:
size mismatch for model.layers.0.input_layernorm.weight: copying a param with shape torch.Size([5120]) from checkpoint, the shape in current model is torch.Size([4096]).
```

#### Why This Occurs in Core ComfyUI ([`comfy/sd.py`](../../../../../../comfy/sd.py)):
```python
# Upstream ComfyUI detection code:
if "model.visual.deepstack_merger_list.0.norm.weight" in sd:  # Standard Hugging Face keys
    return TEModel.QWEN3VL_4B if sd["model.visual.merger.linear_fc2.weight"].shape[0] == 2560 else TEModel.QWEN3VL_8B
```
1. Core ComfyUI checks `shape[0]` of the visual merger. If `shape[0] == 2560` it returns 4B; otherwise, it **unconditionally assumes 8B** (`hidden_size=4096`).
2. For 32B (`shape[0] == 5120`), it misidentifies the architecture as 8B, initializes a 4096-wide model, and crashes when loading 5120-wide parameters.
3. Furthermore, stock `CLIPLoader` lacks native support for Hugging Face `compressed-tensors` Marlin INT4 packed tensors.

### The Custom In-Memory Adapter Solution (`comfyui_minimax_h3_awq_loader.py`)
Rather than maintaining a duplicate modified checkpoint on disk, an in-memory adapter node (**`MiniMaxH3AWQEncoderLoader`**) was authored:

```
                  IN-MEMORY ADAPTATION ARCHITECTURE
                  
  [ Single Checkpoint on Disk (18.99 GB) ]
  • Hugging Face Namespace (model.language_model.*, model.visual.*)
  • Full 64 Layers in compressed-tensors INT32 packing
                    │
                    ▼  MiniMaxH3AWQEncoderLoader
  [ In-Memory Adaptation & Validation ]
  1. Zero-Copy View Repack: Views int32 storage as packed int8 bytes.
  2. Scale Transposition: Materializes flat (K/group, N) layouts for CUDA.
  3. Layer Truncation: Drops layers 50–63 in memory (64 -> 50 layers).
  4. Dynamic Activation Cast: Casts FP32 tokens -> BF16 across W4A16 GEMV.
  5. Source Preprocessing: Executes official video/image processor configs.
                    │
                    ▼
  [ Native ComfyUI H3 Conditioning Pipeline & 33B DiT Generator ]
```

1. **Zero-Copy View Repack:** On little-endian hosts, casting `int32` storage to `int8` yields 4 consecutive bytes containing the exact 2 unsigned nibbles expected by `comfy-kitchen`'s CUDA operator (`gemv_awq_w4a16`).
2. **Scale Transposition:** Materializes $(K/\text{group}, N)$ flat layouts for CUDA consumption without touching disk.
3. **In-Memory 50-Layer Truncation:** Truncates layers 50–63 in memory to match the native 50-layer H3 conditioning contract.
4. **Dynamic Activation Casting:** Casts incoming FP32 activations from `SDClipModel` to BF16 for the CUDA GEMV kernel, then restores FP32 for residual arithmetic, keeping memory footprint strictly at ~14.97 GB staged.
5. **Source-Config Driven Preprocessing:** Dynamically runs `video_preprocessor_config.json` and `image_processor` settings for duration-aware video patching.

---

## 7. Artifact Deliverables & Release Ecosystem

### 1. Hugging Face Model Repository
* **Repository:** [`fbjr/qwen3-vl-32b-W4A16-AWQ-H3`](https://huggingface.co/fbjr/qwen3-vl-32b-W4A16-AWQ-H3)
* **Checkpoints:**
  * Multi-shard distribution (`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`) for vLLM & SGLang serving.
  * Single-file distribution (`qwen3vl_32b_minimax_h3_w4a16_awq.safetensors`, 18.99 GB) for ComfyUI.
  * Full tokenizer and processor configs (`tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`).

### 2. Standalone ComfyUI Extension & Workflows
* **Standalone Loader Node:** `h3_awq_encoder.py` (deleted 2026-09-13 with the lane; a standalone copy survives as `workflows/comfyui_minimax_h3_awq_loader.py` beside this file) (drop-in custom node for ComfyUI).
* **Ready-to-Run Workflow Templates:**
  * `comfyui_minimax_h3_awq_text_to_video.json`: High-speed text-to-video workflow.
  * `comfyui_minimax_h3_awq_image_reference.json`: Omni-reference image conditioning (`Ref2VA`).
  * `comfyui_minimax_h3_awq_first_frame.json`: First/last frame anchor interpolation (`FL2VA`).

### 3. Open Source Git Repository
* **Repository:** `git@github.com:fblissjr/llm-compressor.git` (`main` branch)
* **Codebase Path:** [`examples/multimodal_vision/minimax_h3/`](https://github.com/fblissjr/llm-compressor/tree/main/examples/multimodal_vision/minimax_h3)
  * `quantize_qwen3_vl_32b.py`: Production quantization pipeline.
  * `extract_metadata.py`: Multi-threaded MP4 workflow & prompt metadata extractor.
  * `convert_to_comfyui.py`: Consolidated single-file `.safetensors` builder.
  * `validate.py` & `test_layer50_drift.py`: End-to-end model QA and Layer 50 hidden state benchmarking suite.
  * `upload_to_hf.py`: Automated Hugging Face repository publisher.
